import os
import numpy as np
from typing import Dict

from xml.dom import minidom

from xml.etree.ElementTree import Element, SubElement, tostring


class OpenDriveBuilder:
    def __init__(self):
        pass
    
    def _pretty_xml(self, element: Element) -> str:
            rough_string = tostring(element, encoding="utf-8")
            return minidom.parseString(rough_string).toprettyxml(indent="  ")
    
        
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