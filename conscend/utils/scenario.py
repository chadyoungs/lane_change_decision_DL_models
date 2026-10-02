from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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