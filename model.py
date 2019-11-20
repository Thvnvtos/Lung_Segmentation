import torch
import torch.nn as nn
import torch.nn.functional as F


# Work on Middle Layer and output layer, then construct the model


class ConvUnit(nn.Module):
  """
    Convolution Unit- no Maxpool
    for  now : Conv3D -> BatchNorm -> ReLu * 2 times
    Try modifying to Residual convolutions
  """

  def __init__(self, in_channels, out_channels):
    super().init()
    self.double_conv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size = 3),
        nn.BatchNorm3d(out_channels),
        # inplace=True means it changes the input directly, input is lost
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size = 3),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
      )

  def forward(self,x):
    return self.double_conv(x)


class EncoderUnit(nn.Module):
  """
    An Encoder Unit with the ConvUnit and MaxPool
  """
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.convUnit = ConvUnit(in_channels,out_channels)
    self.maxpool = nn.MaxPool3d(2)
  def forward(self, x):
    skip_conn = self.convUnit(x)
    return skip_conn, self.maxpool(skip_conn)

class DecoderUnit(nn.Module):
  """
    ConvUnit and convTranspose3D

  """
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.convUnit = ConvUnit(in_channels,out_channels)
    self.convTranspose = nn.ConvTranspose3d(out_channels,out_channels,
        kernel_size = 2, stride = 2)

  def forward(self, x1, x2):
    x = torch.cat([x2, x1], dim = 4)  # Verify on runtime need change
    x = self.convUnit(x)
    return self.convTranspose(x)





