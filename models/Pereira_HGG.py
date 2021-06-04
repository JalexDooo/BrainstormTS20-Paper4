import torch as t
import torch.nn as nn
import math

class P_HGG(nn.Module):
    """
    input shape: [bz, 4, 33, 33]
    2D model to predict the class of the center point (预测图像中心点分类)
    patch size: [33, 33]
    Note that out_put = 5
    """
    def __init__(self, config, in_data=4, out_data=5):
        super(P_HGG, self).__init__()
        self.config = config
        self.ly123 = nn.Sequential(
            nn.Conv2d(in_data, 64, kernel_size=3, stride=1),
            nn.RReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.RReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.RReLU(inplace=True)
        )
        self.ly4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.ly567 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.RReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.RReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.RReLU(inplace=True)
        )
        self.ly8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.position_encoding = nn.Parameter(t.zeros(1, 128)) # [1, 256, 768]
        self.fc1 = nn.Sequential(
            nn.Linear(6172, 256), # 1024*6=6144 + 128=6272
            nn.RReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.RReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 5),
            nn.Dropout(0.1),
            nn.Softmax(5)
        )


    def forward(self, x):
        x = self.ly123(x)
        x = self.ly4(x)
        x = self.ly567(x)
        x = self.ly8(x)
        x = x.view(1, -1)
        print(x.shape)
        print(self.position_encoding.shape)
        x = t.cat(x, self.position_encoding)
        # x = self.fc1(x)
        return x