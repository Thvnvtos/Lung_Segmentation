import numpy as np
import pickle
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn

import dataset, model


SMOOTH = 1e-6

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1)
    labels = labels.squeeze(1).int()
    print(labels.type())
    intersection = (outputs & labels).float().sum((1,2,3))
    union = (outputs | labels).float().sum((1,2,3))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    return thresholded




labelled_path = "/powerai/data/retina-unet/data/labelled.pickle"

with open(labelled_path, "rb") as f:
  list_scans = pickle.load(f)

st_scans = [s.split('/')[1] for s in list_scans]

st_scans = st_scans[:50]
scans_path = "/powerai/data/retina-unet/data/LIDC-IDRI_1-100"
masks_path = "/powerai/data/retina-unet/data/lung_masks_LUNA16"

params = {"batch_size": 1, "shuffle": True, "num_workers": 8}

dataset = dataset.Dataset(st_scans, scans_path, masks_path)
data_gen = data.DataLoader(dataset, **params)

device = torch.device("cuda:0")
unet = model.UNet(1,1,8).to(device)

criterion = nn.MSELoss()
optimizer = optim.RMSprop(unet.parameters(), lr = 0.01, weight_decay=1e-8)

for epoch in range(20):
    
    running_loss = 0
    for batch, labels in data_gen:
    
        print(batch.type())
        print(labels.type())
        print("############################## \n \n \n")
        batch = batch.to(device)
        labels = batch.to(device).int()
        
        optimizer.zero_grad()
        preds = unet(batch).float()
        print(preds.type())
        loss = criterion(preds, labels.float())
        loss.requires_grad=True
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(running_loss)
    print("epoch : ", epoch)
    print(running_loss/50)
