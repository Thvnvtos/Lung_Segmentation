import numpy as np
import nrrd, os, scipy.ndimage
from glob import glob
import torch
from torch.utils import data

from . import utils

class Dataset(data.Dataset):
  """
    list_scans is a list containing the filenames of scans
    scans_path and masks_path are the paths of the folders containing the data
    mode : 2d will return slices
  """
  def __init__(self, list_scans, scans_path, masks_path, mode = "3d", scan_size = [128, 256, 256], n_classes = 1):
    self.list_scans = list_scans
    self.scans_path = scans_path
    self.masks_path = masks_path
    self.mode = mode
    self.scan_size = scan_size
    self.n_classes = n_classes

  def __len__(self):
    return len(self.list_scans)

  def __getitem__(self, index):

    scan = self.list_scans[index]

    #load scan and mask
    path = os.path.join(self.scans_path, scan, '*', '*')
    scan_dicom_id = os.path.basename(glob(path)[0])   # used to find the corresponding lung mask
    nrrd_scan = nrrd.read(glob(os.path.join(path, "*CT.nrrd"))[0])   # tuple containing the CT scan and some metadata
    ct_scan = np.swapaxes(nrrd_scan[0], 0, 2)
    seg_mask, _, _ = utils.load_itk(os.path.join(self.masks_path, scan_dicom_id + ".mhd"))# function uses SimpleITK to load lung masks from mhd/zraw data

    if self.n_classes == 3:
      seg_mask[seg_mask == 3] = 1
      seg_mask[seg_mask == 4] = 2
      seg_mask[seg_mask == 5] = 3
    else:
      seg_mask[seg_mask == 5] = 0
      seg_mask[seg_mask > 0] = 1
      
  
    if self.mode == "3d":
      ct_scan = scipy.ndimage.interpolation.zoom(ct_scan, [self.scan_size[0]/float(len(ct_scan)) , self.scan_size[1]/512., self.scan_size[2]/512.], mode = "nearest")
      seg_mask = scipy.ndimage.interpolation.zoom(seg_mask, [self.scan_size[0]/float(len(seg_mask)) , self.scan_size[1]/512., self.scan_size[2]/512.], mode = "nearest")

    if self.mode == "2d":
      return ct_scan[:, np.newaxis, :], seg_mask[:, np.newaxis, :]
    else:
      return ct_scan[np.newaxis, :], seg_mask[np.newaxis, :]

