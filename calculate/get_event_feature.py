# Note: Referenced and modified from https://github.com/chitianhao/lane-change-prediction-lstm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from configs.constant import *
import random
import pickle
from read.read_data import *
from modules.lane import LaneInfo
from modules.feature import Feature

import numpy as np


def time_series_process(input_series, option):
    """
    mean               | 
    standard deviation |             
    median             | 
    25% percentile     | 
    75% percentile     | 
    minimum            | 
    maximum            |
    """
    vehicle_count = 1 if option == "ego_dop" else 7
    
    output_matrix = np.zeros((vehicle_count, 7, 8))

    sur_vehicle_name_list = ["preceding_vehicle", "leftpreceding_vehicle", "rightpreceding_vehicle", "leftfollowing_vehicle", \
                             "rightfollowing_vehicle", "leftalongside_vehicle", "rightalongside_vehicle"]
            
    
    if option == "ego_dop":
        input_series = np.array(input_series)
        for i in range(input_series.shape[1]):
            vector = input_series[:, i]
            mean, std_dev, median, tf_pen, sf_pen, min_, max_ = np.mean(vector), np.std(vector), np.median(vector), \
                                                                np.percentile(vector, 25), np.percentile(vector, 75), \
                                                                np.min(vector), np.max(vector)
            output_matrix[0, :, i] = mean, std_dev, median, tf_pen, sf_pen, min_, max_
    else:
        for idx, sur_vehicle_name in enumerate(sur_vehicle_name_list):
            input_series_per_vehicle = []
            for ele in input_series:
                input_series_per_vehicle.append(tuple(ele[sur_vehicle_name].values()))
            
            input_series_per_vehicle = np.array(input_series_per_vehicle)
            
            for i in range(input_series_per_vehicle.shape[1]):
                vector = input_series_per_vehicle[:, i]
                mean, std_dev, median, tf_pen, sf_pen, min_, max_ = np.mean(vector), np.std(vector), np.median(vector), \
                                                                    np.percentile(vector, 25), np.percentile(vector, 75), \
                                                                    np.min(vector), np.max(vector)
                output_matrix[idx, :, i] = mean, std_dev, median, tf_pen, sf_pen, min_, max_
    
    return output_matrix


class FeatureConstruction(object):
    def __init__(self, hdd_file_number):
        self.hdd_file_number = hdd_file_number
        self.feature_option = FEATURE_CHOICE

        self.tracks_csv = read_tracks_csv(f"{DATASET_ROOT}/data/{self.hdd_file_number}_tracks.csv")
        self.tracks_meta_csv = read_tracks_meta(f"{DATASET_ROOT}/data/{self.hdd_file_number}_tracksMeta.csv")

        self._get_lane_change_keep_ids()
        self.lane_change_result, self.lane_keep_result, self.result = [], [], []
        self.left_lane_change_count, self.right_lane_change_count = 0, 0
        
    def _get_lane_change_keep_ids(self):
        # figure out the lane changing cars and lane keeping cars
        self.lane_changing_ids, self.lane_keeping_ids = [], []
        for key in self.tracks_meta_csv:
            if(self.tracks_meta_csv[key][NUMBER_LANE_CHANGES] > 0):
                self.lane_changing_ids.append(key)
            else:
                self.lane_keeping_ids.append(key)
        print(f"lane change counts: {len(self.lane_changing_ids)}")
        print(f"lane keep counts: {len(self.lane_keeping_ids)}")
    
    def _get_lanechange_frames(self, laneinfo, i):
        # for each car:
        last_boundary = 0
        # list of (starting index, ending index, direction)
        changing_tuple_list = []
        
        # 1. determine the frame we want to use
        for frame_num in range(1, len(self.tracks_csv[i][FRAME])):
            if self.tracks_csv[i][LANE_ID][frame_num] != self.tracks_csv[i][LANE_ID][frame_num-1]:
                # get the direction
                original_lane = self.tracks_csv[i][LANE_ID][frame_num-1]
                new_lane = self.tracks_csv[i][LANE_ID][frame_num]
                direction = LaneInfo._determine_change_direction(laneinfo.lane_num, original_lane, new_lane)
                
                # calculate the starting frame
                starting_change = frame_num - 1
                while starting_change > last_boundary:
                    if LaneInfo._detect_lane_change(laneinfo.lanes_info[original_lane], self.tracks_csv[i][Y][starting_change], laneinfo.lane_width, self.tracks_meta_csv[i][HEIGHT]):
                        break
                    starting_change -= 1

                # calculate the starting and ending frame
                if FRAME_BEFORE_FLAG:
                    starting_point = starting_change - FRAME_TAKEN - FRAME_BEFORE
                    ending_point = starting_change - FRAME_BEFORE
                else:
                    starting_point = starting_change - FRAME_TAKEN
                    ending_point = starting_change
                if starting_point > last_boundary:
                    changing_tuple_list.append(
                        (starting_point, ending_point, direction))
                last_boundary = frame_num
        
        return changing_tuple_list

    def _get_features(self, lane_info_object, vehicle_id, frame_num, original_lane, option):
        feature = Feature(lane_info_object, self.tracks_meta_csv, self.tracks_csv, vehicle_id, frame_num, original_lane)
        
        if self.feature_option == "Normal":
            feature.construct_feature_A()
        elif self.feature_option == "CNN_FC" and option == "ego_dop":
            feature.construct_feature_B(option)
        elif self.feature_option == "CNN_FC" and option == "sur_dop":
            feature.construct_feature_B(option)
        elif self.feature_option == "CNN_FC" and option == "ego_vector":
            feature.construct_feature_B_vector()
        else:
            pass
        
        return feature.ret

    def construct(self):
        self.construct_lanechange_features()
        self.lane_change_count = len(self.lane_change_result)

        if len(self.lane_keeping_ids) >= self.lane_change_count * LANE_CHANGE_KEEP_RATIO:
            # make the lane keeping size the same as lane changing
            self.lane_keeping_ids = random.sample(self.lane_keeping_ids, int(self.lane_change_count*LANE_CHANGE_KEEP_RATIO))
        elif len(self.lane_keeping_ids) < self.lane_change_count:
            raise ValueError("No enough lane keep data")

        self.construct_lanekeep_features()
        self.lane_keep_count = len(self.lane_keep_result)

        self.result = self.lane_change_result + self.lane_keep_result

    def construct_lanechange_features(self):
        laneinfo = LaneInfo(self.hdd_file_number)

        for i in self.lane_changing_ids:
            changing_tuple_list = self._get_lanechange_frames(laneinfo, i)
            
            if not changing_tuple_list:continue
            # add those frames' features
            for pair in changing_tuple_list:
                # for each lane change instance
                cur_change_ego_dop, cur_change_sur_dop = [], []
                start_idx, end_idx, direction = pair
                fail = False

                original_lane = self.tracks_csv[i][LANE_ID][start_idx]
                
                # continue for out of boundary cases
                if original_lane not in laneinfo.lanes_info:
                    continue
                
                try:
                    for frame_num in range(start_idx, end_idx):
                        # construct the object
                        cur_change_ego_dop.append(self._get_features(
                            laneinfo, i, frame_num, original_lane, "ego_dop"))
                        cur_change_sur_dop.append(self._get_features(
                            laneinfo, i, frame_num, original_lane, "sur_dop"))
                    
                    # lane change timestamp occurence
                    cur_change_ego_vector = self._get_features(
                            laneinfo, i, frame_num, original_lane, "ego_vector")
                except Exception as error:
                    #print(error)
                    fail = True
                    break
                
                if not fail:
                    # get the event feature
                    cur_change_event = {"ego_dop": time_series_process(cur_change_ego_dop, "ego_dop"),
                                        "sur_dop": time_series_process(cur_change_sur_dop, "sur_dop"),
                                        "ego_vector": np.array(cur_change_ego_vector)}

                    # add to the result
                    self.lane_change_result.append((cur_change_event, direction))
                    if direction == 1:
                        self.left_lane_change_count += 1
                    else:
                        self.right_lane_change_count += 1
    
    def construct_lanekeep_features(self):
        laneinfo = LaneInfo(self.hdd_file_number)

        for i in self.lane_keeping_ids:
            cur_keep_ego_dop, cur_keep_sur_dop = [], []
            original_lane = self.tracks_csv[i][LANE_ID][0]
            fail = False

            try:
                # notice that FRAME_CONSIST
                for frame_num in range(1, FRAME_TAKEN+FRAME_CONSIST+1):
            
                    feature_ego_dop = self._get_features(
                            laneinfo, i, frame_num, original_lane, "ego_dop")
                    feature_sur_dop = self._get_features(
                            laneinfo, i, frame_num, original_lane, "sur_dop")
                    if frame_num <= FRAME_TAKEN:
                        cur_keep_ego_dop.append(feature_ego_dop)
                        cur_keep_sur_dop.append(feature_sur_dop)

                cur_change_ego_vector = self._get_features(
                        laneinfo, i, FRAME_TAKEN, original_lane, "ego_vector")
            except Exception as error:
                #print(error)
                # handle exception where the total frame is less than FRAME_TAKEN
                fail = True
                break
            
            if not fail:
                cur_keep_event = {"ego_dop": time_series_process(cur_keep_ego_dop, "ego_dop"),
                                  "sur_dop": time_series_process(cur_keep_sur_dop, "sur_dop"),
                                  "ego_vector": np.array(cur_change_ego_vector) }
                
                self.lane_keep_result.append((cur_keep_event, 0))


def cnn_fc_main():
    if not os.path.exists("output"):
        os.makedirs("output")

    total_change, total_keep = 0, 0
    whole_data = []
    for i in range(1, 61):
        number = "{0:0=2d}".format(i)
        feature_construction= FeatureConstruction(number)
        feature_construction.construct()
        result, change_num, keep_num = feature_construction.result, feature_construction.lane_change_count, feature_construction.lane_keep_count
        
        print(f"{i}th file has been processed")
        total_change += change_num
        print("total changes:", total_change)
        total_keep += keep_num
        print("total keeps:", total_keep)
        print("\n")

        whole_data.extend(result)

    whole_data_size = len(whole_data)
    
    random.shuffle(whole_data)
    training_set = whole_data[:int(whole_data_size*0.8)]
    testing_set = whole_data[int(whole_data_size*0.8):]
    
    for subset_name, subset in zip(["training", "testing"], [training_set, testing_set]):
        filename = f"output/result_{subset_name}_{FEATURE_CHOICE}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(subset, f)
            print("Successfully write to:", filename)


if __name__ == "__main__":
    cnn_fc_main()

