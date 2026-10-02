import torch
import torch.nn as nn
    

class CN_FC(nn.Module):
    def __init__(self):
        super(CN_FC, self).__init__()
        self.conv_surround = nn.Sequential(
            nn.Conv2d(7, 16, 4),
            nn.Conv2d(16, 32, 4),
            nn.Flatten()
        )
        self.conv_ego = nn.Sequential(
            nn.Conv2d(1, 16, 4),
            nn.Conv2d(16, 8, 4),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(90, 50),
            nn.Linear(50, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 3),
        )
    
    def forward(self, ego_dop_input, sur_dop_input, ego_vector_input):
        CN_ego_output = self.conv_ego(ego_dop_input)
        CN_surround_output = self.conv_surround(sur_dop_input)
        
        CN_output = torch.cat([CN_ego_output, CN_surround_output, ego_vector_input], dim=1)
        
        output = self.fc(CN_output)
    
        return output

