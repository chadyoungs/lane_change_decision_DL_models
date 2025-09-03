import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DataSet
import time

from model import CN_FC
torch.manual_seed(1)

# with GPU
if torch.cuda.device_count() > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)
net_path = f"{PRE_DIR}/output/best_accuracy_net.pth"

# set hyper-parameters
batch_size = 16
num_epoch = 100

# model, loss_function and optimizer
model = CN_FC()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model.to(device))

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# get the dataloader
train_loader = DataLoader(DataSet("training"), shuffle=True, num_workers=4, batch_size=batch_size)
test_loader = DataLoader(DataSet("testing"), shuffle=True, num_workers=4, batch_size=batch_size)

accuracy_best = 0

for epoch in range(1, num_epoch+1):
    start = time.time()
    total_num = 0
    total_loss = 0.0
    for batch_idx, (ego_dop_data, sur_dop_data, ego_vector_data, labels) in enumerate(train_loader):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        optimizer.zero_grad()

        # Step 2. Run our forward pass.
        if torch.cuda.device_count() > 1:
            ego_dop_data = ego_dop_data.float().to(device)
            sur_dop_data = sur_dop_data.float().to(device)
            ego_vector_data = ego_vector_data.float().to(device)
        tag_scores = model(ego_dop_data, sur_dop_data, ego_vector_data)
        
        # Step 3. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        if torch.cuda.device_count() > 1:
            labels = labels.to(device)
        loss = loss_function(tag_scores, labels)
        total_loss += loss.item()
        total_num += batch_size
        
        loss.backward()
        optimizer.step()

    end = time.time()
    print("\nEpoch", epoch, " run time:", end-start)
    print("loss:", total_loss/total_num)

    # See what the scores are after training
    correct, total = 0, 0
    miss, false_alarm = 0, 0
    
    with torch.no_grad():
        for batch_idx, (ego_dop_data, sur_dop_data, ego_vector_data, labels) in enumerate(test_loader):
            if torch.cuda.device_count() > 1:
                ego_dop_data = ego_dop_data.float().to(device)
                sur_dop_data = sur_dop_data.float().to(device)
                ego_vector_data = ego_vector_data.float().to(device)
                labels = labels.to(device)

            output_scores = model(ego_dop_data, sur_dop_data, ego_vector_data)
            for output_score, label in zip(output_scores, labels):
                _, idx = output_score.max(0)

                if label != 0 and idx == 0:
                    miss += 1
                if label == 0 and idx != 0:
                    false_alarm += 1
                if idx == label:
                    correct += 1
                total += 1
        
        accuracy = correct/total
        if accuracy > accuracy_best:
            accuracy_best = accuracy
            torch.save(model, net_path)
        
        print("testing miss rate:", miss / total)
        print("testing false alarm rate:", false_alarm / total)
        print("testing accuracy:", accuracy)
    
print("best accuray:", accuracy_best)

