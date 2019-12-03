import numpy as np
import pickle, json
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

import model
from data import *


with open("config.json") as f:
  config = json.load(f)

device = torch.device("cuda:0")

with open(config["path"]["labelled_list"], "rb") as f:
  list_scans = pickle.load(f)

st_scans = [s.split('/')[1] for s in list_scans]
st_scans = st_scans[:30]

dataset = dataset.Dataset(st_scans, config["path"]["scans"], config["path"]["masks"], mode = "3d")
unet = model.UNet(1,1, config["train"]["start_filters"]).to(device)

criterion = utils.dice_loss
optimizer = optim.Adam(unet.parameters(), lr = config["train"]["lr"])

scans_per_batch = config["train"]["scans_per_batch"]
slices_per_batch = config["train"]["slices_per_batch"]
neg = config["train"]["neg_examples_per_batch"]

for epoch in range(config["train"]["epochs"]):
  epoch_loss = 0
  for i in range(0, len(dataset), scans_per_batch):
    batch_loss = 0
    batch = np.array([dataset.__getitem__(j)[0] for j in range(i, i+scans_per_batch)]).astype(np.float16)
    labels = np.array([dataset.__getitem__(j)[1] for j in range(i, i+scans_per_batch)]).astype(np.float16)

    batch = torch.Tensor(batch).to(device)
    labels = torch.Tensor(labels).to(device)
    batch.requires_grad = True
    labels.requires_grad = True

    optimizer.zero_grad()
    logits = unet(batch)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print("Batch mean loss : ", loss.item()/scans_per_batch)
    epoch_loss += loss.item()/scans_per_batch
  print("=========> Epoch {} : {}".format(epoch+1, epoch_loss/(len(dataset)/scans_per_batch)))
  if epoch_loss < 0.04:
    break

torch.save(unet.state_dict(), "./model")
