import os
import pickle

import torch.utils.data as data

class DataSet(data.Dataset):
    def __init__(self, option, root="../calculate/output"):
        # testing, training
        self.option = option
        self.data_file = os.path.join(root, f"result_{self.option}_CNN_FC.pickle")

        with open(self.data_file, "rb") as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0]["ego_dop"], self.data[index][0]["sur_dop"], self.data[index][0]["ego_vector"], self.data[index][1]
