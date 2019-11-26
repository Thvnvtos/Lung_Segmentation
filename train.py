import numpy as np
import pickle
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

import dataset, model


def dice_loss(output, labels, eps=1e-7):
  '''
    output, labels shape : [B, 1, Z, Y, X]
  '''
  num = 2. * torch.sum(output * labels)
  denom = torch.sum(output**2 + labels**2)
  return 1 - torch.mean(num / (denom + eps)) 




labelled_path = "/home/ilyass/retina_unet/list_good_v2.pickle"

with open(labelled_path, "rb") as f:
  list_scans = pickle.load(f)

st_scans = [s.split('/')[1] for s in list_scans]

st_scans = st_scans[:50]
scans_path = "/home/ilyass/retina_unet/LIDC_IDRI_1-1012(no238-584)"
masks_path = "/home/ilyass/retina_unet/seg-lungs-LUNA16"

params = {"batch_size": 1, "shuffle": True, "num_workers": 8}

dataset = dataset.Dataset(st_scans, scans_path, masks_path)
data_gen = data.DataLoader(dataset, **params)

device = torch.device("cuda:0")
unet = model.UNet(1,1,8).to(device)

criterion = nn.MSELoss()
optimizer = optim.RMSprop(unet.parameters(), lr = 0.001, weight_decay=1e-8)

for epoch in range(20):
    
    running_loss = 0
    for batch, labels in data_gen:
    
        batch = batch.to(device)
        labels = labels.float().to(device)
        labels.requires_grad = True
        optimizer.zero_grad()
        preds = unet(batch)
        #print("######################### Preds : ")
        #print(preds.shape)
        #print(preds.type())
        #print(preds.max())
        #print(preds.min())
        #print("######################### labels : ")
        #print(labels.shape)
        #print(labels.type())
        #print(preds.max())
        #print(preds.min())
        #print("\n\n##########\n")
        loss = dice_loss(labels, preds)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print("here : " ,loss.item())
    print("epoch : ", epoch)
    print(running_loss/50)
