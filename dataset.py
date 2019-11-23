import numpy as np
import nrrd, os
from glob import glob
import torch
from torch.utils import data

from utils import load_itk

class Dataset(data.Dataset):
  """
    list_scans is a list containing the filenames of scans
    scans_path and masks_path are the paths of the folders containing the data
  """
  def __init__(self, list_scans, scans_path, masks_path):
    self.list_scans = list_scans
    self.scans_path = scans_path
    self.masks_path = masks_path
  
  def __len__(self):
    return len(self.list_scans)

  def __getitem__(self, index):

    scan = self.list_scans[index]

    #load scan and mask
    path = os.path.join(self.scans_path, scan, '*', '*')
    scan_dicom_id = os.path.basename(glob(path)[0])   # used to find the corresponding lung mask 
    nrrd_scan = nrrd.read(glob(os.path.join(path, "*CT.nrrd"))[0])   # tuple containing the CT scan and some metadata
    seg_mask, _, _ = load_itk(os.path.join(self.masks_path, scan_dicom_id + ".mhd"))  # function uses SimpleITK to load lung masks from mhd/zraw data

    ct_scan = nrrd_scan[0]
    mask = seg_mask

    return np.swapaxes(ct_scan, 0, 2), mask

