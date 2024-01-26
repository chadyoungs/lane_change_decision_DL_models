# frame rate 25
FEATURE_CHOICE = "Normal"
LANE_CHANGE_KEEP_RATIO = 1
FRAME_TAKEN = 50  # number of states to construct features, 50 means 2 seconds
FRAME_BEFORE = 25  # frame taken before the lane change, 25 means 1 second
FRAME_CONSIST = 300 # frame requirement before the lane keep, 300 means 12 seconds
FRAME_BEFORE_FLAG = False # use FRAME_BEFORE or not
SAFETIME_HEADWAY = 1.5