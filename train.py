import numpy as np
import pickle
import torch

import dataset, model

labelle_path = "../list_good_v2.pickle"

with open("../list_good_v2.pickle", "rb") as f:
  list_scans = pickle.load(f)

st_scans = [s.split('/')[1] for s in list_scans]

st_scans = list_scans[:200]
scans_path = "/home/ilyass/retina_unet/LIDC_IDRI_1-1012(no238-584)"
masks_path = "/home/ilyass/retina_unet/seg-lungs-LUNA16"

data = dataset.Dataset(list_scans, scans_path, masks_path)
x,y = data.__getitem__(1)

device = torch.device("cuda:0")
unet = model.UNet(1,1,8).to(device)
x = x[np.newaxis,:]
xx = np.array([x])
print(xx.shape)

y = unet(torch.tensor(xx[:,:,90:106,:,:]).to(device))
print(y.shape)

