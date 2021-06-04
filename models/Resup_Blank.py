import torch as t
import torch.nn as nn
import math

from .utils import SingleConvBlock, ResConvBlock_Blank, ConvUpBlock


class ResUp_Blank(nn.Module):
    def __init__(self, config, in_data=4, out_data=4):
        super(ResUp_Blank, self).__init__()
        kn = [32, 64, 128, 256]

        self.in_model = SingleConvBlock(in_data, 16)

        self.ly1 = nn.Sequential(
            ResConvBlock_Blank(16),
            SingleConvBlock(16, kn[0]),
            nn.MaxPool3d(2, 2, 0)
        )
        self.ly2 = nn.Sequential(
            ResConvBlock_Blank(kn[0]),
            SingleConvBlock(kn[0], kn[1]),
            nn.MaxPool3d(2, 2, 0)
        )
        self.ly3 = nn.Sequential(
            ResConvBlock_Blank(kn[1]),
            SingleConvBlock(kn[1], kn[2]),
            nn.MaxPool3d(2, 2, 0)
        )
        self.ly4 = nn.Sequential(
            ResConvBlock_Blank(kn[2]),
            SingleConvBlock(kn[2], kn[3]),
            nn.MaxPool3d(2, 2, 0)
        )

        self.up1 = ConvUpBlock(kn[3], kn[2])
        self.up2 = ConvUpBlock(kn[2], kn[1])
        self.up3 = ConvUpBlock(kn[1], kn[0])
        self.up4 = ConvUpBlock(kn[0], 16)

        self.out_model = SingleConvBlock(16, out_data)


        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        x1 = self.in_model(x)
        x2 = self.ly1(x1)
        x3 = self.ly2(x2)
        x4 = self.ly3(x3)
        bottom = self.ly4(x4)
        x4 = self.up1(bottom, x4)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        x1 = self.up4(x2, x1)
        x = self.out_model(x1)
        return x

