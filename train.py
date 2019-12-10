import numpy as np
import pickle, json
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

import model
from data import *

torch.backends.cudnn.benchmark = True

with open("config.json") as f:
  config = json.load(f)

device = torch.device("cuda:0")

with open(config["path"]["labelled_list"], "rb") as f:
  list_scans = pickle.load(f)

st_scans = [s.split('/')[1] for s in list_scans]

if config["mode"] == "3d":
  train_scans = st_scans[:config["train3d"]["train_size"]]
  val_scans = st_scans[config["train3d"]["train_size"]:]
  train_data = dataset.Dataset(train_scans, config["path"]["scans"], config["path"]["masks"], mode="3d", scan_size = config["train3d"]["scan_size"], n_classes = config["train3d"]["n_classes"])
  val_data = dataset.Dataset(val_scans, config["path"]["scans"], config["path"]["masks"], mode = "3d", scan_size = config["train3d"]["scan_size"])
  unet = model.UNet(1,config["train3d"]["n_classes"], config["train3d"]["start_filters"], bilinear = False).to(device)
  criterion = utils.dice_loss
  optimizer = optim.Adam(unet.parameters(), lr = config["train3d"]["lr"])
  batch_size = config["train3d"]["batch_size"]
  epochs = config["train3d"]["epochs"]
  val_steps = config["train3d"]["validation_steps"]
  val_size = config["train3d"]["validation_size"]
else:
  st_scans = st_scans[:config["train2d"]["train_size"]]
  dataset = dataset.Dataset(st_scans, config["path"]["scans"], config["path"]["masks"], mode = "2d")
  unet = model.UNet(1,1, config["train2d"]["start_filters"], bilinear = True).to(device)
  criterion = utils.dice_loss
  optimizer = optim.Adam(unet.parameters(), lr = config["train2d"]["lr"])
  batch_size = config["train2d"]["batch_size"]
  slices_per_batch = config["train2d"]["slices_per_batch"]
  neg = config["train2d"]["neg_examples_per_batch"]
  epochs = config["train2d"]["epochs"]

best_val_loss = 1e16
for epoch in range(epochs):
  epoch_loss = 0
  for i in range(0, len(train_data), batch_size):
    batch_loss = 0
    batch = np.array([train_data.__getitem__(j)[0] for j in range(i, i+batch_size)]).astype(np.float16)
    labels = np.array([train_data.__getitem__(j)[1] for j in range(i, i+batch_size)]).astype(np.float16)

    batch = torch.Tensor(batch).to(device)
    labels = torch.Tensor(labels).to(device)
    batch.requires_grad = True
    labels.requires_grad = True

    optimizer.zero_grad()
    logits = unet(batch)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print("Epoch {} ==> Batch {} mean loss : {}".format(epoch+1, (i+1)%(val_steps), loss.item()/batch_size))
    epoch_loss += loss.item()/batch_size
    del batch
    del labels
    torch.cuda.empty_cache()
    if (i+1)%val_steps == 0:
      print("===================> Calculating validation loss ... ")
      ids = np.random.randint(0, len(val_data), val_size)
      val_loss = 0
      for scan_id in ids:
        batch = np.array([val_data.__getitem__(j)[0] for j in range(scan_id, scan_id+batch_size)]).astype(np.float16)
        labels = np.array([val_data.__getitem__(j)[1] for j in range(scan_id, scan_id+batch_size)]).astype(np.float16)
        batch = torch.Tensor(batch).to(device)
        labels = torch.Tensor(labels).to(device)
        logits = unet(batch)
        loss = criterion(logits, labels)
        val_loss += loss.item()
      val_loss /= val_size
      print("\n # Validation Loss : ", val_loss)
      if val_loss < best_val_loss:
        print("\nSaving Better Model... ")
        torch.save(unet.state_dict(), "./model")
        best_val_loss = val_loss
      print("\n")

