# Note: Referenced and modified from https://github.com/chitianhao/lane-change-prediction-lstm
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import OrderedDict
from configs.config import *
from configs.constant import *
from read.read_data import *
from lane import LaneInfo


class Feature(object):
    '''
        Feature A:
        Construct all the features for the RNN to train:
        Here is the list:
        Difference of the ego car's Y position and the lane center: ΔY
        Ego car's X velocity: Vx
        Ego car's Y velocity: Vy
        Ego car's X acceleration: Ax
        Ego car's Y acceleration: Ay
        Ego car type: T
        TTC of preceding car: TTCp
        TTC of following car: TTCf
        TTC of left preceding car: TTClp
        TTC of left alongside car: TTCla
        TTC of left following car: TTClf
        TTC of right preceding car: TTCrp
        TTC of right alongside car: TTCra
        TTC of right following car: TTCrf

        Feature B:
        Construct all the features for the CNN-ego to train:
        Here is the list:
        ___________________  Relative Local x/y | Lateral/Longitudinal Velocity | Lateral/Longitudinal Acceleration | Space/Time Headway | 
        mean               |                    |                               |                                   |                    |
        standard deviation |                    |                               |                                   |                    |
        median             |                    |                               |                                   |                    |  
        25% percentile     |                    |                               |                                   |                    |
        75% percentile     |                    |                               |                                   |                    |
        minimum            |                    |                               |                                   |                    |
        maximum            |                    |                               |                                   |                    |

        ,
        Construct all the features for the CNN-surrounding (P, PL, PR, FL, FR, ASL, ASR) to train:
        Here is the list:
        ___________________  Relative Local x/y | Lateral/Longitudinal Velocity | Lateral/Longitudinal Acceleration | Space/Time Headway | 
        mean               |                    |                               |                                   |                    |
        standard deviation |                    |                               |                                   |                    |
        median             |                    |                               |                                   |                    |  
        25% percentile     |                    |                               |                                   |                    |
        75% percentile     |                    |                               |                                   |                    |
        minimum            |                    |                               |                                   |                    |
        maximum            |                    |                               |                                   |                    |
        ,
        (10-dim vector)
        V_ego - V_preceding
        V_precedingleft - V_preceding
        V_precedingright - V_preceding
        D_precedingleft - D_preceding
        D_precedingright - D_preceding
        D_followingleft
        D_followingright
        V_ego - V_followingleft
        V_ego - V_followingright
        D_preceding - Vego * T_h (safetime headway T_h)
    '''
    def __init__(self, lane_info_object, track_metadata_csv, tracks_csv, vehicle_id, frame_num, original_lane):
        self.lanes_info = lane_info_object.lanes_info
        self.track_metadata_csv = track_metadata_csv
        
        self.tracks_csv = tracks_csv
        self.vehicle_id = vehicle_id
        self.frame_num = frame_num
        self.original_lane = original_lane

        self.lane_num = lane_info_object.lane_num

        self.cur_feature = OrderedDict()
    
    @staticmethod
    def _calculate_ttc(track_metadata_csv_ego, track_metadata_csv_target, tracks_csv_ego, tracks_csv_target, going, frame_num):
        """
        Calculate time to collision of target car and current car
        """
        target_frame = track_metadata_csv_ego[INITIAL_FRAME] + \
            frame_num - track_metadata_csv_target[INITIAL_FRAME]
        target_x = tracks_csv_target[X][target_frame]
        cur_x = tracks_csv_ego[X][frame_num]
        target_v = tracks_csv_target[X_VELOCITY][target_frame]
        cur_v = tracks_csv_ego[X_VELOCITY][frame_num]

        if target_v == cur_v:
            return 99999
        
        if going == 1:
            # going left (up)
            if cur_x > target_x:
                ttc = (cur_x - target_x) / (cur_v - target_v)
            else:
                ttc = (target_x - cur_x) / (target_v - cur_v)
        else:
            # going right (down)
            if cur_x > target_x:
                ttc = (cur_x - target_x) / (target_v - cur_v)
            else:
                ttc = (target_x - cur_x) / (cur_v - target_v)
        
        if ttc < 0:
            return 99999
        else:
            return ttc
            
    def construct_feature_A(self):
        """
        for https://github.com/chitianhao/lane-change-prediction-lstm
        to be extended to another feature construction method
        """
        self.cur_feature["left_lane_exist"], self.cur_feature["right_lane_exist"] = LaneInfo._determine_lane_exist(self.lane_num,
            self.original_lane)
        
        # We need to consider the fact that right/left are different for top/bottom lanes.
        # top lanes are going left      <----
        # bottom lanes are going right  ---->
        # left -> negative, right -> positive
        going = 0  # 1 left, 2 right
        if self.lane_num == 4:
            if self.original_lane in [2, 3]:
                going = 1
            else:
                going = 2
        else:
            if self.original_lane in [2, 3, 4, 5]:
                going = 1
            else:
                going = 2

        if going == 1:
            self.cur_feature["delta_y"] = self.tracks_csv[self.vehicle_id][Y][self.frame_num] - \
                self.lanes_info[self.original_lane]  # up
            self.cur_feature["y_velocity"] = -self.tracks_csv[self.vehicle_id][Y_VELOCITY][self.frame_num]
            self.cur_feature["y_acceleration"] = - \
                self.tracks_csv[self.vehicle_id][Y_ACCELERATION][self.frame_num]
        else:
            self.cur_feature["delta_y"] = self.lanes_info[self.original_lane] - \
                self.tracks_csv[self.vehicle_id][Y][self.frame_num]  # down
            self.cur_feature["y_velocity"] = self.tracks_csv[self.vehicle_id][Y_VELOCITY][self.frame_num]
            self.cur_feature["y_acceleration"] = self.tracks_csv[self.vehicle_id][Y_ACCELERATION][self.frame_num]

        self.cur_feature["x_velocity"] = self.tracks_csv[self.vehicle_id][X_VELOCITY][self.frame_num]
        self.cur_feature["x_acceleration"] = self.tracks_csv[self.vehicle_id][X_ACCELERATION][self.frame_num]
        self.cur_feature["car_type"] = 1 if self.track_metadata_csv[self.vehicle_id][CLASS] == "Car" else -1

        # surrounding cars info
        preceding_vehicle_id = self.tracks_csv[self.vehicle_id][PRECEDING_ID][self.frame_num]
        
        self.cur_feature["preceding_ttc"] = self._calculate_ttc(
            self.track_metadata_csv[self.vehicle_id], self.track_metadata_csv[preceding_vehicle_id],
            self.tracks_csv[self.vehicle_id], self.tracks_csv[preceding_vehicle_id], going, self.frame_num) if preceding_vehicle_id != 0 else 999
        
        following_vehicle_id = self.tracks_csv[self.vehicle_id][FOLLOWING_ID][self.frame_num]
        self.cur_feature["following_ttc"] = self._calculate_ttc(
            self.track_metadata_csv[self.vehicle_id], self.track_metadata_csv[following_vehicle_id],
            self.tracks_csv[self.vehicle_id], self.tracks_csv[following_vehicle_id], going, self.frame_num) if following_vehicle_id != 0 else 999
        
        left_preceding_vehicle_id = self.tracks_csv[self.vehicle_id][LEFT_PRECEDING_ID][self.frame_num]
        self.cur_feature["left_preceding_ttc"] = self._calculate_ttc(
            self.track_metadata_csv[self.vehicle_id], self.track_metadata_csv[left_preceding_vehicle_id],
            self.tracks_csv[self.vehicle_id], self.tracks_csv[left_preceding_vehicle_id], going, self.frame_num) if left_preceding_vehicle_id != 0 else 999
        
        leftalongside_vehicle_id = self.tracks_csv[self.vehicle_id][LEFT_ALONGSIDE_ID][self.frame_num]
        self.cur_feature["left_alongside_ttc"] = self._calculate_ttc(
            self.track_metadata_csv[self.vehicle_id], self.track_metadata_csv[leftalongside_vehicle_id],
            self.tracks_csv[self.vehicle_id], self.tracks_csv[leftalongside_vehicle_id], going, self.frame_num) if leftalongside_vehicle_id != 0 else 999
        
        leftfollowing_vehicle_id = self.tracks_csv[self.vehicle_id][LEFT_FOLLOWING_ID][self.frame_num]
        self.cur_feature["left_following_ttc"] = self._calculate_ttc(
            self.track_metadata_csv[self.vehicle_id], self.track_metadata_csv[leftfollowing_vehicle_id],
            self.tracks_csv[self.vehicle_id], self.tracks_csv[leftfollowing_vehicle_id], going, self.frame_num) if leftfollowing_vehicle_id != 0 else 999
        
        rightpreceding_vehicle_id = self.tracks_csv[self.vehicle_id][RIGHT_PRECEDING_ID][self.frame_num]
        self.cur_feature["right_preceding_ttc"] = self._calculate_ttc(
            self.track_metadata_csv[self.vehicle_id], self.track_metadata_csv[rightpreceding_vehicle_id],
            self.tracks_csv[self.vehicle_id], self.tracks_csv[rightpreceding_vehicle_id], going, self.frame_num) if rightpreceding_vehicle_id != 0 else 999
        
        rightalongside_vehicle_id = self.tracks_csv[self.vehicle_id][RIGHT_ALONGSIDE_ID][self.frame_num]
        self.cur_feature["right_alongside_ttc"] = self._calculate_ttc(
            self.track_metadata_csv[self.vehicle_id], self.track_metadata_csv[rightalongside_vehicle_id],
            self.tracks_csv[self.vehicle_id], self.tracks_csv[rightalongside_vehicle_id], going, self.frame_num) if rightalongside_vehicle_id != 0 else 999
        
        rightfollowing_vehicle_id = self.tracks_csv[self.vehicle_id][RIGHT_FOLLOWING_ID][self.frame_num]
        self.cur_feature["right_following_ttc"] = self._calculate_ttc(
            self.track_metadata_csv[self.vehicle_id], self.track_metadata_csv[rightfollowing_vehicle_id],
            self.tracks_csv[self.vehicle_id], self.tracks_csv[rightfollowing_vehicle_id], going, self.frame_num) if rightfollowing_vehicle_id != 0 else 999

        self.ret = tuple(self.cur_feature.values())
    
    def construct_feature_B(self, option):
        """
        option: ego_dop, sur_dop
        Relative Local x/y
        Lateral/Longitudinal Velocity
        Lateral/Longitudinal Acceleration
        Space/Time Headway
        """
        if option == "ego_dop":
            preceding_vehicle_id = self.tracks_csv[self.vehicle_id][PRECEDING_ID][self.frame_num]

            self.cur_feature["rel_x"] = self.tracks_csv[self.vehicle_id][X][self.frame_num] - self.tracks_csv[self.vehicle_id][X][0]
            self.cur_feature["rel_y"] = self.tracks_csv[self.vehicle_id][Y][self.frame_num] - self.tracks_csv[self.vehicle_id][X][0]
            self.cur_feature["lateral_Vel"] = self.tracks_csv[self.vehicle_id][X_VELOCITY][self.frame_num]
            self.cur_feature["longi_Vel"] = self.tracks_csv[self.vehicle_id][Y_VELOCITY][self.frame_num]
            self.cur_feature["lateral_Acc"] = self.tracks_csv[self.vehicle_id][X_ACCELERATION][self.frame_num]
            self.cur_feature["longi_Acc"] = self.tracks_csv[self.vehicle_id][Y_ACCELERATION][self.frame_num]
            self.cur_feature["space_headway"] =  self.tracks_csv[preceding_vehicle_id][X][self.frame_num] - \
                self.tracks_csv[self.vehicle_id][X][self.frame_num] if preceding_vehicle_id != 0 else 999
            self.cur_feature["time_headway"] = self.cur_feature["space_headway"] / self.tracks_csv[self.vehicle_id][X_VELOCITY][self.frame_num] \
                if preceding_vehicle_id != 0 else 999
            
            self.ret = tuple(self.cur_feature.values())
            
        elif option == "sur_dop":
            preceding_vehicle_id = self.tracks_csv[self.vehicle_id][PRECEDING_ID][self.frame_num]
            leftpreceding_vehicle_id = self.tracks_csv[self.vehicle_id][LEFT_PRECEDING_ID][self.frame_num]
            rightpreceding_vehicle_id = self.tracks_csv[self.vehicle_id][RIGHT_PRECEDING_ID][self.frame_num]
            leftfollowing_vehicle_id = self.tracks_csv[self.vehicle_id][LEFT_FOLLOWING_ID][self.frame_num]
            rightfollowing_vehicle_id = self.tracks_csv[self.vehicle_id][RIGHT_FOLLOWING_ID][self.frame_num]
            leftalongside_vehicle_id = self.tracks_csv[self.vehicle_id][LEFT_ALONGSIDE_ID][self.frame_num]
            rightalongside_vehicle_id = self.tracks_csv[self.vehicle_id][RIGHT_ALONGSIDE_ID][self.frame_num]
            
            sur_vehicle_id_list = [preceding_vehicle_id, leftpreceding_vehicle_id, rightpreceding_vehicle_id, leftfollowing_vehicle_id, \
                                   rightfollowing_vehicle_id, leftalongside_vehicle_id, rightalongside_vehicle_id]
            sur_vehicle_name_list = ["preceding_vehicle", "leftpreceding_vehicle", "rightpreceding_vehicle", "leftfollowing_vehicle", \
                                     "rightfollowing_vehicle", "leftalongside_vehicle", "rightalongside_vehicle"]
            
            for sur_vehicle_id, sur_vehicle in zip(sur_vehicle_id_list, sur_vehicle_name_list):
                self.cur_feature[sur_vehicle] = OrderedDict()
                
                if sur_vehicle_id != 0:
                    sur_preceding_vehicle_id = self.tracks_csv[sur_vehicle_id][PRECEDING_ID][self.frame_num]

                    self.cur_feature[sur_vehicle]["rel_x"] = self.tracks_csv[sur_vehicle_id][X][self.frame_num] - self.tracks_csv[sur_vehicle_id][X][0]
                    self.cur_feature[sur_vehicle]["rel_y"] = self.tracks_csv[sur_vehicle_id][Y][self.frame_num] - self.tracks_csv[sur_vehicle_id][X][0]
                    self.cur_feature[sur_vehicle]["lateral_Vel"] = self.tracks_csv[sur_vehicle_id][X_VELOCITY][self.frame_num]
                    self.cur_feature[sur_vehicle]["longi_Vel"] = self.tracks_csv[sur_vehicle_id][Y_VELOCITY][self.frame_num]
                    self.cur_feature[sur_vehicle]["lateral_Acc"] = self.tracks_csv[sur_vehicle_id][X_ACCELERATION][self.frame_num]
                    self.cur_feature[sur_vehicle]["longi_Acc"] = self.tracks_csv[sur_vehicle_id][Y_ACCELERATION][self.frame_num]
                    self.cur_feature[sur_vehicle]["space_headway"] =  self.tracks_csv[sur_preceding_vehicle_id][X][self.frame_num] - \
                        self.tracks_csv[sur_vehicle_id][X][self.frame_num] if sur_preceding_vehicle_id != 0 else 999
                    self.cur_feature[sur_vehicle]["time_headway"] = self.cur_feature[sur_vehicle]["space_headway"] / self.tracks_csv[sur_vehicle_id][X_VELOCITY][self.frame_num] \
                        if sur_preceding_vehicle_id != 0 else 999
                else:
                    self.cur_feature[sur_vehicle]["rel_x"] = 0
                    self.cur_feature[sur_vehicle]["rel_y"] = 0
                    self.cur_feature[sur_vehicle]["lateral_Vel"] = 0
                    self.cur_feature[sur_vehicle]["longi_Vel"] = 0
                    self.cur_feature[sur_vehicle]["lateral_Acc"] = 0
                    self.cur_feature[sur_vehicle]["longi_Acc"] = 0
                    self.cur_feature[sur_vehicle]["space_headway"] = 999
                    self.cur_feature[sur_vehicle]["time_headway"] = 999

            self.ret = self.cur_feature

    def construct_feature_B_vector(self):
        # ego vector
        preceding_vehicle_id = self.tracks_csv[self.vehicle_id][PRECEDING_ID][self.frame_num]
        leftpreceding_vehicle_id = self.tracks_csv[self.vehicle_id][LEFT_PRECEDING_ID][self.frame_num]
        rightpreceding_vehicle_id = self.tracks_csv[self.vehicle_id][RIGHT_PRECEDING_ID][self.frame_num]
        leftfollowing_vehicle_id = self.tracks_csv[self.vehicle_id][LEFT_FOLLOWING_ID][self.frame_num]
        rightfollowing_vehicle_id = self.tracks_csv[self.vehicle_id][RIGHT_FOLLOWING_ID][self.frame_num]
        
        dis_preceding = self.tracks_csv[preceding_vehicle_id][X][self.frame_num] - \
            self.tracks_csv[self.vehicle_id][X][self.frame_num] if preceding_vehicle_id != 0 else 999
        dis_leftpreceding = self.tracks_csv[leftpreceding_vehicle_id][X][self.frame_num] - \
            self.tracks_csv[self.vehicle_id][X][self.frame_num] if leftpreceding_vehicle_id != 0 else 999
        dis_rightpreceding = self.tracks_csv[rightpreceding_vehicle_id][X][self.frame_num] - \
            self.tracks_csv[self.vehicle_id][X][self.frame_num] if rightpreceding_vehicle_id != 0 else 999
        dis_leftfollowing = self.tracks_csv[self.vehicle_id][X][self.frame_num] - \
            self.tracks_csv[leftfollowing_vehicle_id][X][self.frame_num] if leftfollowing_vehicle_id != 0 else 999
        dis_rightfollowing = self.tracks_csv[self.vehicle_id][X][self.frame_num] - \
            self.tracks_csv[rightfollowing_vehicle_id][X][self.frame_num] if rightfollowing_vehicle_id != 0 else 999

        self.cur_feature["ego_preceding_vel_diff"] = self.tracks_csv[self.vehicle_id][X_VELOCITY][self.frame_num] - \
            self.tracks_csv[preceding_vehicle_id][X_VELOCITY][self.frame_num] if preceding_vehicle_id != 0 else 0
        self.cur_feature["precedingleft_vel_diff"] = self.tracks_csv[leftpreceding_vehicle_id][X_VELOCITY][self.frame_num] - \
            self.tracks_csv[preceding_vehicle_id][X_VELOCITY][self.frame_num] if (preceding_vehicle_id != 0 and leftpreceding_vehicle_id != 0) else 0
        self.cur_feature["precedingright_vel_diff"] = self.tracks_csv[rightpreceding_vehicle_id][X_VELOCITY][self.frame_num] - \
            self.tracks_csv[preceding_vehicle_id][X_VELOCITY][self.frame_num] if (preceding_vehicle_id != 0 and rightpreceding_vehicle_id != 0) else 0
        self.cur_feature["dis_precedingleft_preceding"] = dis_leftpreceding - dis_preceding
        self.cur_feature["dis_precedingright_preceding"] = dis_rightpreceding - dis_preceding
        self.cur_feature["followingleft_vel_diff"] = self.tracks_csv[self.vehicle_id][X_VELOCITY][self.frame_num] - \
            self.tracks_csv[leftfollowing_vehicle_id][X_VELOCITY][self.frame_num] if leftfollowing_vehicle_id != 0 else 0
        self.cur_feature["followingright_vel_diff"] = self.tracks_csv[self.vehicle_id][X_VELOCITY][self.frame_num] - \
            self.tracks_csv[rightfollowing_vehicle_id][X_VELOCITY][self.frame_num] if rightfollowing_vehicle_id != 0 else 0
        self.cur_feature["dis_followingleft"] = dis_leftfollowing
        self.cur_feature["dis_followingright"] = dis_rightfollowing
        self.cur_feature["safe"] = dis_preceding - self.tracks_csv[self.vehicle_id][X_VELOCITY][self.frame_num] * SAFETIME_HEADWAY

        self.ret = tuple(self.cur_feature.values())