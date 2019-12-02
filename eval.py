import numpy as np
import pickle, nrrd, json
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
st_scans = st_scans[70:]

dataset = dataset.Dataset(st_scans, config["path"]["scans"], config["path"]["masks"])

criterion = utils.dice_loss
unet = model.UNet(1,1, config["train"]["start_filters"]).to(device)
unet.load_state_dict(torch.load("./model"))

for i in range(len(dataset)):
    X,y = dataset.__getitem__(i)
    xx = np.squeeze(X)
    X = torch.Tensor(X.astype(np.float16)).to(device)
    y = torch.Tensor(y.astype(np.float16)).to(device)
    mask = []
    for j in range(len(X)):
        logits = unet(X[j:j+1])
        logits[logits < 0.5] = 0
        logits[logits >= 0.5] = 1
        mask.append(logits.cpu().detach().numpy())
        loss = criterion(logits, y[j:j+1])
        print(loss.item())
    mask = np.concatenate(mask)
    mask = np.squeeze(mask)
    nrrd.write("ct.nrrd",xx)
    nrrd.write("mask.nrrd",mask)
    break
