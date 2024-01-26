# Note: Referenced and modified from https://github.com/chitianhao/lane-change-prediction-lstm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.config import *
from configs.constant import *
from read.read_data import *


class LaneInfo(object):
    def __init__(self, hdd_file_number):
        self.hdd_file_number = hdd_file_number
        self.recording_meta = read_recording_meta(f"{DATASET_ROOT}/data/{self.hdd_file_number}_recordingMeta.csv")
        self.lane_num = len(self.recording_meta[UPPER_LANE_MARKINGS]) + len(self.recording_meta[LOWER_LANE_MARKINGS]) - 2

        self.lanes_info = {}
        self._lane_info()
    
    def _lane_info(self):
        if self.lane_num == 4:
            # 4 lanes
            self.lanes_info[2] = self.recording_meta[UPPER_LANE_MARKINGS][0]
            self.lanes_info[3] = self.recording_meta[UPPER_LANE_MARKINGS][1]
            self.lanes_info[5] = self.recording_meta[LOWER_LANE_MARKINGS][0]
            self.lanes_info[6] = self.recording_meta[LOWER_LANE_MARKINGS][1]
            self.lane_width = ((self.lanes_info[3] - self.lanes_info[2]) + 
                               (self.lanes_info[6] - self.lanes_info[5])) / 2
        elif self.lane_num == 6:
            # 6 lanes
            self.lanes_info[2] = self.recording_meta[UPPER_LANE_MARKINGS][0]
            self.lanes_info[3] = self.recording_meta[UPPER_LANE_MARKINGS][1]
            self.lanes_info[4] = self.recording_meta[UPPER_LANE_MARKINGS][2]
            self.lanes_info[6] = self.recording_meta[LOWER_LANE_MARKINGS][0]
            self.lanes_info[7] = self.recording_meta[LOWER_LANE_MARKINGS][1]
            self.lanes_info[8] = self.recording_meta[LOWER_LANE_MARKINGS][2]
            self.lane_width = ((self.lanes_info[3] - self.lanes_info[2]) + (self.lanes_info[4] - self.lanes_info[3]) + 
                               (self.lanes_info[7] - self.lanes_info[6]) + (self.lanes_info[8] - self.lanes_info[7])) / 4
        elif self.lane_num == 7:
            # 7 lanes: track 58 ~ 60
            self.lanes_info[2] = self.recording_meta[UPPER_LANE_MARKINGS][0]
            self.lanes_info[3] = self.recording_meta[UPPER_LANE_MARKINGS][1]
            self.lanes_info[4] = self.recording_meta[UPPER_LANE_MARKINGS][2]
            self.lanes_info[5] = self.recording_meta[UPPER_LANE_MARKINGS][3]
            self.lanes_info[7] = self.recording_meta[LOWER_LANE_MARKINGS][0]
            self.lanes_info[8] = self.recording_meta[LOWER_LANE_MARKINGS][1]
            self.lanes_info[9] = self.recording_meta[LOWER_LANE_MARKINGS][2]
            self.lane_width = ((self.lanes_info[3] - self.lanes_info[2]) + (self.lanes_info[4] - self.lanes_info[3]) + 
                               (self.lanes_info[5] - self.lanes_info[4]) + (self.lanes_info[8] - self.lanes_info[7]) + 
                               (self.lanes_info[9] - self.lanes_info[8])) / 5
    
    @staticmethod
    def _determine_lane_exist(lane_num, cur_lane):
        '''
        return: left_exist, right_exist 
        Have to do this shit in a hardcoded way to determine the existence of neighbor lanes.
        '''
        if lane_num == 4:
            if cur_lane == 2 or cur_lane == 6:
                return 1, 0
            else:
                return 0, 1
        elif lane_num == 6:
            if cur_lane == 2 or cur_lane == 8:
                return 1, 0
            elif cur_lane == 3 or cur_lane == 7:
                return 1, 1
            else:
                return 0, 1
        elif lane_num == 7:
            if cur_lane == 2 or cur_lane == 9:
                return 1, 0
            elif cur_lane == 3 or cur_lane == 4 or cur_lane == 8:
                return 1, 1
            else:
                return 0, 1
    
    @staticmethod
    def _detect_lane_change(lane_center, cur_y, lane_width, car_height):
        delta_y = abs(lane_center - cur_y)
        relative_diff = delta_y / car_height
        if(relative_diff < 0.5):
            return True
        else:
            return False
    
    @staticmethod
    def _determine_change_direction(lane_num, ori_laneId, new_laneId):
        '''
        return 1 upon left change
        return 2 upon right change
        '''
        if lane_num == 4:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 6 and new_laneId == 5):
                return 1
            else:
                return 2
        else:
            # left:
            if (ori_laneId == 2 and new_laneId == 3) or (ori_laneId == 4 and new_laneId == 5) \
                or (ori_laneId == 3 and new_laneId == 4) or (ori_laneId == 7 and new_laneId == 6) \
                    or (ori_laneId == 8 and new_laneId == 7) or (ori_laneId == 9 and new_laneId == 8):
                return 1
            else:
                return 2