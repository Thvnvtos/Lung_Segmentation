import numpy as np
import torch
import SimpleITK as sitk

# Not sure if works for all format (Tested only on mhd/zraw format)
def load_itk(filename):
  itkimage = sitk.ReadImage(filename)
  ct_scan = sitk.GetArrayFromImage(itkimage)
  origin = np.array(list(reversed(itkimage.GetOrigin())))
  spacing = np.array(list(reversed(itkimage.GetSpacing())))
  return ct_scan, origin, spacing


def dice_loss(logits, labels, eps=1e-7):
  '''
    logits, labels, shape : [B, 1, Y, X]

  '''
  num = 2. * torch.sum(logits * labels)
  denom = torch.sum(logits**2 + labels**2)
  return 1 - torch.mean(num / (denom + eps))
