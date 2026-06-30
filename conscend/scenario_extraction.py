import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.constant import DATASET_ROOT
from read.read_data import (
    FRAME,
    LANE_ID,
    LEFT_ALONGSIDE_ID,
    LEFT_FOLLOWING_ID,
    LOWER_LANE_MARKINGS,
    PRECEDING_ID,
    RIGHT_ALONGSIDE_ID,
    RIGHT_FOLLOWING_ID,
    THW,
    UPPER_LANE_MARKINGS,
    X,
    X_ACCELERATION,
    X_VELOCITY,
    Y,
    Y_VELOCITY,
    read_recording_meta,
    read_tracks_csv,
    read_tracks_meta,
)


FRAME_RATE = 25
MAX_ALKS_SPEED_MPS = 60 / 3.6
FOLLOWING_DISTANCE_TIME_GAP_S = 1.6
LANE_CHANGE_LOOKBACK_FRAMES = 18
LANE_CHANGE_LOOKAHEAD_FRAMES = 25
ADJACENT_LOOKBACK_FRAMES = 10
CAR_FOLLOWING_RATIO = 0.6
FREE_DRIVING_RATIO = 0.7
FREE_DRIVING_THW_S = 3.0
EMERGENCY_BRAKE_ACCELERATION_MPS2 = -2.0


@dataclass
class ScenarioRecord:
    scenario_id: str
    recording_id: str
    ego_vehicle_id: int
    scenario_type: str
    start_frame: int
    end_frame: int
    event_frame: int
    direction: str
    lane_id: int
    target_lane_id: int
    openx_lane_id: int
    openx_target_lane_id: int
    ego_speed_mps: float
    ego_x_m: float
    ego_y_m: float
    thw_s: float
    required_following_distance_m: float
    peak_lateral_velocity_mps: float
    longitudinal_acceleration_mps2: float
    adjacent_vehicle_id: int
    preceding_vehicle_id: int
    lane_change_duration_s: float
    lane_width_m: float
    road_file: str
    actors: List[Dict[str, float]]
    trajectory: List[Dict[str, float]]


class ConScenDExtractor:
    def __init__(self, dataset_root: str = DATASET_ROOT, output_root: Optional[str] = None):
        self.dataset_root = dataset_root
        self.output_root = output_root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "conscend"
        )
        os.makedirs(self.output_root, exist_ok=True)

    def run(self, recordings: Optional[Iterable[int]] = None) -> List[ScenarioRecord]:
        scenarios: List[ScenarioRecord] = []
        recording_ids = recordings if recordings is not None else range(1, 61)

        for recording_index in recording_ids:
            recording_id = f"{recording_index:02d}"
            recording_scenarios = self.extract_recording(recording_id)
            if not recording_scenarios:
                continue
            scenarios.extend(recording_scenarios)

        self._write_summary(scenarios)
        return scenarios

    def extract_recording(self, recording_id: str) -> List[ScenarioRecord]:
        data_dir = os.path.join(self.dataset_root, "data")
        tracks_path = os.path.join(data_dir, f"{recording_id}_tracks.csv")
        tracks_meta_path = os.path.join(data_dir, f"{recording_id}_tracksMeta.csv")
        recording_meta_path = os.path.join(data_dir, f"{recording_id}_recordingMeta.csv")

        if not os.path.exists(tracks_path):
            return []

        tracks = read_tracks_csv(tracks_path)
        read_tracks_meta(tracks_meta_path)
        recording_meta = read_recording_meta(recording_meta_path)
        lane_id_map, lane_width = self._build_lane_mapping(recording_meta)

        road_dir = os.path.join(self.output_root, "roads")
        scenario_dir = os.path.join(self.output_root, "scenarios", recording_id)
        metadata_dir = os.path.join(self.output_root, "metadata", recording_id)
        os.makedirs(road_dir, exist_ok=True)
        os.makedirs(scenario_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        road_file = os.path.join(road_dir, f"{recording_id}_straight_highway.xodr")
        self._write_opendrive_file(road_file, lane_width, recording_meta, lane_id_map)

        scenarios: List[ScenarioRecord] = []
        scenario_counter = 0

        for vehicle_id in sorted(tracks.keys()):
            vehicle_track = tracks[vehicle_id]
            frame_values = vehicle_track[FRAME]
            lane_values = vehicle_track[LANE_ID]
            lane_change_indices = np.where(lane_values[1:] != lane_values[:-1])[0] + 1

            if len(lane_change_indices) == 0:
                scenario = self._extract_non_lane_change_scenario(
                    recording_id, vehicle_id, vehicle_track, lane_id_map, lane_width, road_file
                )
                if scenario is not None:
                    scenario_counter += 1
                    scenario.scenario_id = f"{recording_id}_{scenario_counter:04d}"
                    self._persist_scenario(scenario, scenario_dir, metadata_dir)
                    scenarios.append(scenario)
                continue

            for lane_change_index in lane_change_indices:
                if abs(vehicle_track[X_VELOCITY][lane_change_index]) > MAX_ALKS_SPEED_MPS:
                    continue

                scenario_counter += 1
                scenario = self._extract_lane_change_scenario(
                    recording_id=recording_id,
                    scenario_id=f"{recording_id}_{scenario_counter:04d}",
                    vehicle_id=vehicle_id,
                    vehicle_track=vehicle_track,
                    frame_values=frame_values,
                    lane_change_index=lane_change_index,
                    lane_id_map=lane_id_map,
                    lane_width=lane_width,
                    road_file=road_file,
                    tracks=tracks,
                )
                if scenario is None:
                    scenario_counter -= 1
                    continue

                self._persist_scenario(scenario, scenario_dir, metadata_dir)
                scenarios.append(scenario)

        return scenarios

    def _extract_non_lane_change_scenario(
        self,
        recording_id: str,
        vehicle_id: int,
        vehicle_track: Dict[str, np.ndarray],
        lane_id_map: Dict[int, int],
        lane_width: float,
        road_file: str,
    ) -> Optional[ScenarioRecord]:
        if np.max(np.abs(vehicle_track[X_VELOCITY])) > MAX_ALKS_SPEED_MPS:
            return None

        preceding_presence_ratio = np.mean(vehicle_track[PRECEDING_ID] != 0)
        free_driving_ratio = np.mean(vehicle_track[THW] >= FREE_DRIVING_THW_S)

        if preceding_presence_ratio >= CAR_FOLLOWING_RATIO:
            scenario_type = "car_following_emergency" if np.min(vehicle_track[X_ACCELERATION]) <= EMERGENCY_BRAKE_ACCELERATION_MPS2 else "car_following_comfortable"
        elif free_driving_ratio >= FREE_DRIVING_RATIO:
            scenario_type = "free_driving"
        else:
            return None

        start_index = 0
        end_index = min(len(vehicle_track[FRAME]) - 1, FRAME_RATE * 4)
        start_frame = int(vehicle_track[FRAME][start_index])
        end_frame = int(vehicle_track[FRAME][end_index])
        lane_id = int(vehicle_track[LANE_ID][start_index])
        preceding_vehicle_id = int(vehicle_track[PRECEDING_ID][start_index])

        trajectory = self._collect_trajectory(vehicle_track, start_index, end_index, lane_id_map)
        actors = self._build_actor_list(vehicle_id, preceding_vehicle_id)

        return ScenarioRecord(
            scenario_id="",
            recording_id=recording_id,
            ego_vehicle_id=vehicle_id,
            scenario_type=scenario_type,
            start_frame=start_frame,
            end_frame=end_frame,
            event_frame=start_frame,
            direction="keep",
            lane_id=lane_id,
            target_lane_id=lane_id,
            openx_lane_id=lane_id_map[lane_id],
            openx_target_lane_id=lane_id_map[lane_id],
            ego_speed_mps=float(vehicle_track[X_VELOCITY][start_index]),
            ego_x_m=float(vehicle_track[X][start_index]),
            ego_y_m=float(vehicle_track[Y][start_index]),
            thw_s=float(vehicle_track[THW][start_index]),
            required_following_distance_m=float(abs(vehicle_track[X_VELOCITY][start_index]) * FOLLOWING_DISTANCE_TIME_GAP_S),
            peak_lateral_velocity_mps=float(np.max(np.abs(vehicle_track[Y_VELOCITY][start_index : end_index + 1]))),
            longitudinal_acceleration_mps2=float(np.min(vehicle_track[X_ACCELERATION][start_index : end_index + 1])),
            adjacent_vehicle_id=0,
            preceding_vehicle_id=preceding_vehicle_id,
            lane_change_duration_s=0.0,
            lane_width_m=lane_width,
            road_file=road_file,
            actors=actors,
            trajectory=trajectory,
        )

    def _extract_lane_change_scenario(
        self,
        recording_id: str,
        scenario_id: str,
        vehicle_id: int,
        vehicle_track: Dict[str, np.ndarray],
        frame_values: np.ndarray,
        lane_change_index: int,
        lane_id_map: Dict[int, int],
        lane_width: float,
        road_file: str,
        tracks: Dict[int, Dict[str, np.ndarray]],
    ) -> Optional[ScenarioRecord]:
        if lane_change_index <= 0:
            return None

        lane_id = int(vehicle_track[LANE_ID][lane_change_index - 1])
        target_lane_id = int(vehicle_track[LANE_ID][lane_change_index])
        if lane_id not in lane_id_map or target_lane_id not in lane_id_map:
            return None

        direction = "left" if lane_id_map[target_lane_id] > lane_id_map[lane_id] else "right"
        start_index = max(0, lane_change_index - LANE_CHANGE_LOOKBACK_FRAMES)
        end_index = min(len(frame_values) - 1, lane_change_index + LANE_CHANGE_LOOKAHEAD_FRAMES)
        event_frame = int(frame_values[lane_change_index])
        start_frame = int(frame_values[start_index])
        end_frame = int(frame_values[end_index])

        scenario_type, adjacent_vehicle_id = self._classify_lane_change(vehicle_track, lane_change_index, direction)
        preceding_vehicle_id = int(vehicle_track[PRECEDING_ID][lane_change_index])
        peak_lateral_velocity_mps = float(np.max(np.abs(vehicle_track[Y_VELOCITY][start_index : end_index + 1])))
        trajectory = self._collect_trajectory(vehicle_track, start_index, end_index, lane_id_map)
        actors = self._build_actor_list(vehicle_id, preceding_vehicle_id, adjacent_vehicle_id)

        if adjacent_vehicle_id != 0 and adjacent_vehicle_id in tracks:
            actors.append(self._build_neighbor_actor("adjacent_vehicle", adjacent_vehicle_id, tracks[adjacent_vehicle_id], event_frame))

        return ScenarioRecord(
            scenario_id=scenario_id,
            recording_id=recording_id,
            ego_vehicle_id=vehicle_id,
            scenario_type=scenario_type,
            start_frame=start_frame,
            end_frame=end_frame,
            event_frame=event_frame,
            direction=direction,
            lane_id=lane_id,
            target_lane_id=target_lane_id,
            openx_lane_id=lane_id_map[lane_id],
            openx_target_lane_id=lane_id_map[target_lane_id],
            ego_speed_mps=float(vehicle_track[X_VELOCITY][lane_change_index]),
            ego_x_m=float(vehicle_track[X][lane_change_index]),
            ego_y_m=float(vehicle_track[Y][lane_change_index]),
            thw_s=float(vehicle_track[THW][lane_change_index]),
            required_following_distance_m=float(abs(vehicle_track[X_VELOCITY][lane_change_index]) * FOLLOWING_DISTANCE_TIME_GAP_S),
            peak_lateral_velocity_mps=peak_lateral_velocity_mps,
            longitudinal_acceleration_mps2=float(np.min(vehicle_track[X_ACCELERATION][start_index : end_index + 1])),
            adjacent_vehicle_id=adjacent_vehicle_id,
            preceding_vehicle_id=preceding_vehicle_id,
            lane_change_duration_s=float((end_frame - start_frame) / FRAME_RATE),
            lane_width_m=lane_width,
            road_file=road_file,
            actors=actors,
            trajectory=trajectory,
        )

    def _classify_lane_change(
        self, vehicle_track: Dict[str, np.ndarray], lane_change_index: int, direction: str
    ) -> Tuple[str, int]:
        lookback_start = max(0, lane_change_index - ADJACENT_LOOKBACK_FRAMES)
        if direction == "left":
            following = vehicle_track[LEFT_FOLLOWING_ID][lookback_start:lane_change_index]
            alongside = vehicle_track[LEFT_ALONGSIDE_ID][lookback_start:lane_change_index]
        else:
            following = vehicle_track[RIGHT_FOLLOWING_ID][lookback_start:lane_change_index]
            alongside = vehicle_track[RIGHT_ALONGSIDE_ID][lookback_start:lane_change_index]

        adjacent_candidates = np.concatenate([following, alongside]) if len(following) or len(alongside) else np.array([])
        adjacent_candidates = adjacent_candidates[adjacent_candidates != 0]

        if adjacent_candidates.size > 0:
            return "cut_in", int(adjacent_candidates[0])
        return "simple_lane_change", 0

    def _collect_trajectory(
        self, vehicle_track: Dict[str, np.ndarray], start_index: int, end_index: int, lane_id_map: Dict[int, int]
    ) -> List[Dict[str, float]]:
        trajectory = []
        for current_index in range(start_index, end_index + 1):
            lane_id = int(vehicle_track[LANE_ID][current_index])
            trajectory.append(
                {
                    "frame": int(vehicle_track[FRAME][current_index]),
                    "x_m": float(vehicle_track[X][current_index]),
                    "y_m": float(vehicle_track[Y][current_index]),
                    "lane_id": lane_id,
                    "openx_lane_id": lane_id_map.get(lane_id, lane_id),
                    "speed_mps": float(vehicle_track[X_VELOCITY][current_index]),
                    "lateral_velocity_mps": float(vehicle_track[Y_VELOCITY][current_index]),
                }
            )
        return trajectory

    def _build_actor_list(self, ego_vehicle_id: int, preceding_vehicle_id: int = 0, adjacent_vehicle_id: int = 0) -> List[Dict[str, float]]:
        actors = []
        actors.append({"role": "ego_vehicle", "vehicle_id": ego_vehicle_id})
        if preceding_vehicle_id != 0:
            actors.append({"role": "preceding_vehicle", "vehicle_id": preceding_vehicle_id})
        if adjacent_vehicle_id != 0:
            actors.append({"role": "adjacent_vehicle", "vehicle_id": adjacent_vehicle_id})
        return actors

    def _build_neighbor_actor(
        self, role: str, vehicle_id: int, vehicle_track: Dict[str, np.ndarray], event_frame: int
    ) -> Dict[str, float]:
        frame_index = self._find_frame_index(vehicle_track[FRAME], event_frame)
        return {
            "role": role,
            "vehicle_id": vehicle_id,
            "frame": int(vehicle_track[FRAME][frame_index]),
            "x_m": float(vehicle_track[X][frame_index]),
            "y_m": float(vehicle_track[Y][frame_index]),
            "speed_mps": float(vehicle_track[X_VELOCITY][frame_index]),
        }

    def _build_lane_mapping(self, recording_meta: Dict[str, np.ndarray]) -> Tuple[Dict[int, int], float]:
        upper_count = len(recording_meta[UPPER_LANE_MARKINGS]) - 1
        lower_count = len(recording_meta[LOWER_LANE_MARKINGS]) - 1

        lane_id_map: Dict[int, int] = {}
        current_highd_lane_id = 2
        for lane_offset in range(upper_count):
            lane_id_map[current_highd_lane_id] = lane_offset + 1
            current_highd_lane_id += 1

        current_highd_lane_id += 1
        for lane_offset in range(lower_count):
            lane_id_map[current_highd_lane_id] = -(lane_offset + 1)
            current_highd_lane_id += 1

        lane_widths = []
        upper_markings = recording_meta[UPPER_LANE_MARKINGS]
        lower_markings = recording_meta[LOWER_LANE_MARKINGS]
        if len(upper_markings) > 1:
            lane_widths.extend(np.diff(upper_markings))
        if len(lower_markings) > 1:
            lane_widths.extend(np.diff(lower_markings))
        lane_width = float(np.mean(lane_widths)) if lane_widths else 3.75

        return lane_id_map, lane_width

    def _write_opendrive_file(self, output_path: str, lane_width: float, recording_meta: Dict[str, np.ndarray], lane_id_map: Dict[int, int]) -> None:
        road = Element("OpenDRIVE")
        SubElement(
            road,
            "header",
            {
                "revMajor": "1",
                "revMinor": "4",
                "name": os.path.basename(output_path),
                "version": "1.00",
                "date": "2026-06-30",
                "north": "0",
                "south": "0",
                "east": "0",
                "west": "0",
            },
        )
        road_element = SubElement(road, "road", {"name": "straight_highway", "length": "500.0", "id": "1", "junction": "-1"})
        plan_view = SubElement(road_element, "planView")
        geometry = SubElement(plan_view, "geometry", {"s": "0", "x": "0", "y": "0", "hdg": "0", "length": "500.0"})
        SubElement(geometry, "line")

        lanes = SubElement(road_element, "lanes")
        lane_section = SubElement(lanes, "laneSection", {"s": "0"})
        left = SubElement(lane_section, "left")
        center = SubElement(lane_section, "center")
        right = SubElement(lane_section, "right")
        SubElement(center, "lane", {"id": "0", "type": "none", "level": "false"})

        positive_lane_ids = sorted([lane_id for lane_id in lane_id_map.values() if lane_id > 0])
        negative_lane_ids = sorted([lane_id for lane_id in lane_id_map.values() if lane_id < 0], reverse=True)

        for lane_id in positive_lane_ids:
            lane = SubElement(left, "lane", {"id": str(lane_id), "type": "driving", "level": "false"})
            SubElement(lane, "width", {"sOffset": "0", "a": f"{lane_width:.3f}", "b": "0", "c": "0", "d": "0"})

        for lane_id in negative_lane_ids:
            lane = SubElement(right, "lane", {"id": str(lane_id), "type": "driving", "level": "false"})
            SubElement(lane, "width", {"sOffset": "0", "a": f"{lane_width:.3f}", "b": "0", "c": "0", "d": "0"})

        with open(output_path, "w", encoding="utf-8") as file_object:
            file_object.write(self._pretty_xml(road))

    def _persist_scenario(self, scenario: ScenarioRecord, scenario_dir: str, metadata_dir: str) -> None:
        metadata_path = os.path.join(metadata_dir, f"{scenario.scenario_id}.json")
        scenario_path = os.path.join(scenario_dir, f"{scenario.scenario_id}.xosc")

        with open(metadata_path, "w", encoding="utf-8") as file_object:
            json.dump(asdict(scenario), file_object, indent=2)

        with open(scenario_path, "w", encoding="utf-8") as file_object:
            file_object.write(self._build_openscenario_xml(scenario))

    def _build_openscenario_xml(self, scenario: ScenarioRecord) -> str:
        root = Element("OpenSCENARIO")
        SubElement(
            root,
            "FileHeader",
            {
                "revMajor": "1",
                "revMinor": "0",
                "date": "2026-06-30T00:00:00",
                "description": f"ConScenD-style {scenario.scenario_type} scenario from highD recording {scenario.recording_id}",
                "author": "GitHub Copilot Task Agent",
            },
        )

        parameter_declarations = SubElement(root, "ParameterDeclarations")
        self._add_parameter(parameter_declarations, "egoSpeedInit", "double", f"{scenario.ego_speed_mps:.3f}")
        self._add_parameter(parameter_declarations, "egoStartX", "double", f"{scenario.ego_x_m:.3f}")
        self._add_parameter(parameter_declarations, "egoLaneId", "integer", str(scenario.openx_lane_id))
        self._add_parameter(parameter_declarations, "targetLaneId", "integer", str(scenario.openx_target_lane_id))
        self._add_parameter(parameter_declarations, "requiredFollowingDistance", "double", f"{scenario.required_following_distance_m:.3f}")
        self._add_parameter(parameter_declarations, "peakLateralVelocity", "double", f"{scenario.peak_lateral_velocity_mps:.3f}")

        catalog_locations = SubElement(root, "CatalogLocations")
        SubElement(catalog_locations, "VehicleCatalog", {"directory": "../catalogs/vehicles"})
        SubElement(catalog_locations, "ControllerCatalog", {"directory": "../catalogs/controllers"})

        road_network = SubElement(root, "RoadNetwork")
        SubElement(road_network, "LogicFile", {"filepath": f"../../roads/{os.path.basename(scenario.road_file)}"})

        entities = SubElement(root, "Entities")
        self._add_vehicle_entity(entities, "EgoVehicle")
        if scenario.preceding_vehicle_id != 0:
            self._add_vehicle_entity(entities, "TargetVehicle")

        storyboard = SubElement(root, "Storyboard")
        init = SubElement(storyboard, "Init")
        actions = SubElement(init, "Actions")
        self._add_init_private_action(actions, "EgoVehicle", scenario.openx_lane_id, scenario.ego_x_m, scenario.ego_speed_mps)
        if scenario.preceding_vehicle_id != 0:
            target_speed = max(scenario.ego_speed_mps - 2.0, 0.0)
            target_x = scenario.ego_x_m + max(10.0, scenario.required_following_distance_m)
            self._add_init_private_action(actions, "TargetVehicle", scenario.openx_target_lane_id, target_x, target_speed)

        story = SubElement(storyboard, "Story", {"name": "ConScenDStory"})
        act = SubElement(story, "Act", {"name": "MainAct"})
        act_start_trigger = SubElement(act, "StartTrigger")
        act_condition_group = SubElement(act_start_trigger, "ConditionGroup")
        act_condition = SubElement(act_condition_group, "Condition", {"name": "ActStart", "delay": "0", "conditionEdge": "rising"})
        act_by_value = SubElement(act_condition, "ByValueCondition")
        SubElement(act_by_value, "SimulationTimeCondition", {"value": "0.0", "rule": "greaterThan"})
        maneuver_group = SubElement(act, "ManeuverGroup", {"name": "MainManeuverGroup", "maximumExecutionCount": "1"})
        actors = SubElement(maneuver_group, "Actors", {"selectTriggeringEntities": "false"})
        SubElement(actors, "EntityRef", {"entityRef": "EgoVehicle"})
        maneuver = SubElement(maneuver_group, "Maneuver", {"name": "MainManeuver"})
        event = SubElement(maneuver, "Event", {"name": "PrimaryEvent", "priority": "overwrite"})
        action = SubElement(event, "Action", {"name": "PrimaryAction"})

        if scenario.scenario_type in {"cut_in", "simple_lane_change"}:
            private_action = SubElement(action, "PrivateAction")
            lateral_action = SubElement(private_action, "LateralAction")
            lane_change_action = SubElement(lateral_action, "LaneChangeAction")
            SubElement(
                lane_change_action,
                "LaneChangeActionDynamics",
                {"dynamicsShape": "sinusoidal", "value": f"{max(scenario.lane_change_duration_s, 1.0):.3f}", "dynamicsDimension": "time"},
            )
            lane_change_target = SubElement(lane_change_action, "LaneChangeTarget")
            SubElement(lane_change_target, "AbsoluteTargetLane", {"value": str(scenario.openx_target_lane_id)})
        else:
            private_action = SubElement(action, "PrivateAction")
            longitudinal_action = SubElement(private_action, "LongitudinalAction")
            speed_action = SubElement(longitudinal_action, "SpeedAction")
            SubElement(speed_action, "SpeedActionDynamics", {"dynamicsShape": "linear", "value": "2.0", "dynamicsDimension": "time"})
            speed_target = SubElement(speed_action, "SpeedActionTarget")
            target_speed = (
                scenario.ego_speed_mps
                if scenario.scenario_type in {"free_driving", "car_following_comfortable"}
                else max(scenario.ego_speed_mps - 5.0, 0.0)
            )
            SubElement(speed_target, "AbsoluteTargetSpeed", {"value": f"{target_speed:.3f}"})

        start_trigger = SubElement(event, "StartTrigger")
        condition_group = SubElement(start_trigger, "ConditionGroup")
        condition = SubElement(condition_group, "Condition", {"name": "SimulationStart", "delay": "0", "conditionEdge": "rising"})
        by_value = SubElement(condition, "ByValueCondition")
        SubElement(by_value, "SimulationTimeCondition", {"value": "0.0", "rule": "greaterThan"})

        stop_trigger = SubElement(storyboard, "StopTrigger")
        condition_group = SubElement(stop_trigger, "ConditionGroup")
        condition = SubElement(condition_group, "Condition", {"name": "StopAfterTenSeconds", "delay": "0", "conditionEdge": "rising"})
        by_value = SubElement(condition, "ByValueCondition")
        SubElement(by_value, "SimulationTimeCondition", {"value": "10.0", "rule": "greaterThan"})

        return self._pretty_xml(root)

    def _add_vehicle_entity(self, entities: Element, name: str) -> None:
        scenario_object = SubElement(entities, "ScenarioObject", {"name": name})
        vehicle = SubElement(scenario_object, "Vehicle", {"name": name, "vehicleCategory": "car"})
        SubElement(vehicle, "BoundingBox")
        performance = SubElement(vehicle, "Performance", {"maxSpeed": "70", "maxAcceleration": "8", "maxDeceleration": "9"})
        vehicle.addprevious = performance

    def _add_init_private_action(self, actions: Element, entity_ref: str, lane_id: int, x_position: float, speed: float) -> None:
        private = SubElement(actions, "Private", {"entityRef": entity_ref})
        private_action = SubElement(private, "PrivateAction")
        teleport_action = SubElement(private_action, "TeleportAction")
        position = SubElement(teleport_action, "Position")
        SubElement(position, "LanePosition", {"roadId": "1", "laneId": str(lane_id), "offset": "0", "s": f"{max(x_position, 0.0):.3f}"})
        private_action = SubElement(private, "PrivateAction")
        longitudinal_action = SubElement(private_action, "LongitudinalAction")
        speed_action = SubElement(longitudinal_action, "SpeedAction")
        SubElement(speed_action, "SpeedActionDynamics", {"dynamicsShape": "step", "value": "0", "dynamicsDimension": "time"})
        target = SubElement(speed_action, "SpeedActionTarget")
        SubElement(target, "AbsoluteTargetSpeed", {"value": f"{max(speed, 0.0):.3f}"})

    def _add_parameter(self, parameter_declarations: Element, name: str, parameter_type: str, value: str) -> None:
        SubElement(
            parameter_declarations,
            "ParameterDeclaration",
            {"name": name, "parameterType": parameter_type, "value": value},
        )

    def _pretty_xml(self, element: Element) -> str:
        rough_string = tostring(element, encoding="utf-8")
        return minidom.parseString(rough_string).toprettyxml(indent="  ")

    def _write_summary(self, scenarios: List[ScenarioRecord]) -> None:
        summary_path = os.path.join(self.output_root, "conscend_summary.json")
        scenario_counts: Dict[str, int] = {}
        for scenario in scenarios:
            scenario_counts[scenario.scenario_type] = scenario_counts.get(scenario.scenario_type, 0) + 1

        with open(summary_path, "w", encoding="utf-8") as file_object:
            json.dump(
                {
                    "total_scenarios": len(scenarios),
                    "scenario_counts": scenario_counts,
                    "constraints": {
                        "max_alks_speed_mps": MAX_ALKS_SPEED_MPS,
                        "following_distance_time_gap_s": FOLLOWING_DISTANCE_TIME_GAP_S,
                        "lane_change_lookback_frames": LANE_CHANGE_LOOKBACK_FRAMES,
                        "lane_change_lookahead_frames": LANE_CHANGE_LOOKAHEAD_FRAMES,
                    },
                },
                file_object,
                indent=2,
            )

    def _find_frame_index(self, frame_array: np.ndarray, target_frame: int) -> int:
        matches = np.where(frame_array == target_frame)[0]
        if len(matches) > 0:
            return int(matches[0])

        insertion_point = int(np.searchsorted(frame_array, target_frame))
        if insertion_point <= 0:
            return 0
        if insertion_point >= len(frame_array):
            return len(frame_array) - 1
        previous_frame = frame_array[insertion_point - 1]
        next_frame = frame_array[insertion_point]
        if math.fabs(previous_frame - target_frame) <= math.fabs(next_frame - target_frame):
            return insertion_point - 1
        return insertion_point
