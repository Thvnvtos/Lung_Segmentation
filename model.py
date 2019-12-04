import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvUnit(nn.Module):
  """
    Convolution Unit -
    for  now : (Conv3D -> BatchNorm -> ReLu) * 2
    Try modifying to Residual convolutions
  """

  def __init__(self, in_channels, out_channels):
    super(ConvUnit, self).__init__()
    self.double_conv = nn.Sequential(

        nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True), # inplace=True means it changes the input directly, input is lost

        nn.Conv3d(out_channels, out_channels, kernel_size = 3, padding = 1),
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
    self.encoder = nn.Sequential(
        nn.MaxPool3d(2),
        ConvUnit(in_channels, out_channels)
    )
  def forward(self, x):
    return self.encoder(x)


class DecoderUnit(nn.Module):
  """
    ConvUnit and upsample with Upsample or convTranspose

  """
  def __init__(self, in_channels, out_channels, bilinear=False):
    super().__init__()

    if bilinear:
      # Only for 2D model
      self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    else:
      self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

    self.conv = ConvUnit(in_channels, out_channels)

  def forward(self, x1, x2):

      x1 = self.up(x1)

      diffZ = x2.size()[2] - x1.size()[2]
      diffY = x2.size()[3] - x1.size()[3]
      diffX = x2.size()[4] - x1.size()[4]
      x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

      x = torch.cat([x2, x1], dim=1)
      return self.conv(x)

class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

  def forward(self, x):
    return self.conv(x)




###########   Model :

class UNet(nn.Module):

  def __init__(self, in_channels, n_classes, s_channels, bilinear = False):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.n_classes = n_classes
    self.s_channels = s_channels
    self.bilinear = bilinear

    self.conv = ConvUnit(in_channels, s_channels)
    self.enc1 = EncoderUnit(s_channels, 2 * s_channels)
    self.enc2 = EncoderUnit(2 * s_channels, 4 * s_channels)
    self.enc3 = EncoderUnit(4 * s_channels, 8 * s_channels)
    self.enc4 = EncoderUnit(8 * s_channels, 8 * s_channels)

    self.dec1 = DecoderUnit(16 * s_channels, 4 * s_channels, self.bilinear)
    self.dec2 = DecoderUnit(8 * s_channels, 2 * s_channels, self.bilinear)
    self.dec3 = DecoderUnit(4 * s_channels, s_channels, self.bilinear)
    self.dec4 = DecoderUnit(2 * s_channels, s_channels, self.bilinear)
    self.out = OutConv(s_channels, n_classes)

  def forward(self, x):
    x1 = self.conv(x)
    x2 = self.enc1(x1)
    x3 = self.enc2(x2)
    x4 = self.enc3(x3)
    x5 = self.enc4(x4)

    mask = self.dec1(x5, x4)
    mask = self.dec2(mask, x3)
    mask = self.dec3(mask, x2)
    mask = self.dec4(mask, x1)
    mask = self.out(mask)

    return mask
