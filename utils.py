import numpy as np
import SimpleITK as sitk

# Not sure if works for all format (Tested only on mhd/zraw format)
def load_itk(filename):
  itkimage = sitk.ReadImage(filename)
  ct_scan = sitk.GetArrayFromImage(itkimage)
  origin = np.array(list(reversed(itkimage.GetOrigin())))
  spacing = np.array(list(reversed(itkimage.GetSpacing())))
  return ct_scan, origin, spacing

