import os
import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.parent.absolute().as_posix())
from xml.dom import minidom

from conscend.utils.scenario import ScenarioRecord
from xml.etree.ElementTree import Element, SubElement, tostring

class OpenScenarioBuilder:
    def __init__(self):
        pass

    def _add_vehicle_entity(self, entities: Element, name: str) -> None:
        scenario_object = SubElement(entities, "ScenarioObject", {"name": name})
        vehicle = SubElement(scenario_object, "Vehicle", {"name": name, "vehicleCategory": "car"})
        SubElement(vehicle, "BoundingBox")
        SubElement(vehicle, "Performance", {"maxSpeed": "70", "maxAcceleration": "8", "maxDeceleration": "9"})

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

    def _add_vehicle_entity(self, entities: Element, name: str) -> None:
        scenario_object = SubElement(entities, "ScenarioObject", {"name": name})
        vehicle = SubElement(scenario_object, "Vehicle", {"name": name, "vehicleCategory": "car"})
        SubElement(vehicle, "BoundingBox")
        SubElement(vehicle, "Performance", {"maxSpeed": "70", "maxAcceleration": "8", "maxDeceleration": "9"})

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
                "author": "ConScenDExtractor",
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