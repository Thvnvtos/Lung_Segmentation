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
    super(ConvUnit, self).__init__()
    self.double_conv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 2),
        nn.BatchNorm3d(out_channels),
        # inplace=True means it changes the input directly, input is lost
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 0),
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
    super(EncoderUnit, self).__init__()
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
    self.convUnit = ConvUnit(in_channels,in_channels // 2)
    self.convTranspose = nn.ConvTranspose3d(in_channels // 2, out_channels, kernel_size = 2, stride = 2)

  def forward(self, x1, x2):
    #print("Concatenating : ", x2.shape, x1.shape)
    diffZ = x2.size()[2] - x1.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    diffX = x2.size()[4] - x1.size()[4]
    #print(diffZ, diffY, diffX)
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2,diffZ // 2, diffZ - diffZ // 2])
    x = torch.cat([x2, x1], dim = 1)  # Verify on runtime need change
    x = self.convUnit(x)
    return self.convTranspose(x)

class MiddleUnit(nn.Module):
  """
  """
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.convUnit = ConvUnit(in_channels,in_channels * 2)
    self.convTranspose = nn.ConvTranspose3d(in_channels * 2, out_channels, kernel_size = 2, stride = 2)

  def forward(self, x):
    x = self.convUnit(x)
    x = self.convTranspose(x)
    return x

class OutputUnit(nn.Module):
  """

  """
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.convUnit = ConvUnit(in_channels,in_channels // 2)
    self.outConv = nn.Conv3d(in_channels // 2, out_channels, kernel_size = 1)
  
  def forward(self, x1, x2):
    #print("Concatenating : ", x2.shape, x1.shape)
    x = torch.cat([x2, x1], dim = 1)  # Verify on runtime need change    print(x.shape)
    x = self.convUnit(x)
    return self.outConv(x).int()


###########   Model : 

class UNet(nn.Module):

  def __init__(self, in_channels, n_classes, s_channels):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.n_classes = n_classes
    self.s_channels = s_channels

    self.enc1 = EncoderUnit(in_channels, s_channels)
    self.enc2 = EncoderUnit(s_channels, 2 * s_channels)
    self.enc3 = EncoderUnit(2 * s_channels, 4 * s_channels)
    self.enc4 = EncoderUnit(4 * s_channels, 8 * s_channels)

    self.mid = MiddleUnit(8 * s_channels, 8 * s_channels)

    self.dec1 = DecoderUnit(16 * s_channels, 4 * s_channels)
    self.dec2 = DecoderUnit(8 * s_channels, 2 * s_channels)
    self.dec3 = DecoderUnit(4 * s_channels, s_channels)
    self.out = OutputUnit(2 * s_channels, n_classes)

  def forward(self, x):
    skip_x1, x1 = self.enc1(x)
    #print(x1.shape, skip_x1.shape)
    skip_x2, x2 = self.enc2(x1)
    #print(x2.shape, skip_x2.shape)
    skip_x3, x3 = self.enc3(x2)
    #print(x3.shape, skip_x3.shape)
    skip_x4, x4 = self.enc4(x3)
    #print(x4.shape, skip_x4.shape)

    mid = self.mid(x4)

    mask = self.dec1(mid, skip_x4)
    mask = self.dec2(mask, skip_x3)
    mask = self.dec3(mask, skip_x2)
    mask = self.out(mask, skip_x1)
    return mask
