import numpy as np
import pickle
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F

import dataset, model, utils

scans_path = "/home/ilyass/retina_unet/LIDC_IDRI_1-1012(no238-584)"
masks_path = "/home/ilyass/retina_unet/seg-lungs-LUNA16"
labelled_path = "/home/ilyass/retina_unet/list_good_v2.pickle"

with open(labelled_path, "rb") as f:
  list_scans = pickle.load(f)

st_scans = [s.split('/')[1] for s in list_scans]

st_scans = st_scans[:30]

dataset = dataset.Dataset(st_scans, scans_path, masks_path)

device = torch.device("cuda:0")

unet = model.UNet(1,1, 32).to(device)

criterion = utils.dice_loss
optimizer = optim.Adam(unet.parameters(), lr = 0.001)

batch_size = 1
slices_per_batch = 4

for epoch in range(1):
  epoch_loss = 0
  for i in range(0, len(dataset), batch_size):
    batch_loss = 0
    batch = np.concatenate([dataset.__getitem__(j)[0] for j in range(i, i+batch_size)]).astype(np.float16)
    labels = np.concatenate([dataset.__getitem__(j)[1] for j in range(i, i+batch_size)]).astype(np.float16)

    slices = np.random.randint(0, len(batch), slices_per_batch)

    batch = torch.Tensor(batch[slices]).to(device)
    labels = torch.Tensor(labels[slices]).to(device)
    batch.requires_grad = True
    labels.requires_grad = True

    optimizer.zero_grad()
    logits = unet(batch)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print("Batch mean loss : ", loss.item()/batch_size)
    epoch_loss += loss.item()/batch_size
  print("=========> Epoch {} : {}".format(epoch+1, epoch_loss/(len(dataset)/batch_size)))
