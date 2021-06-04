import torch as t
import torch.nn as nn

import numpy as np
import warnings
import math
import copy

from torch.nn.modules.utils import _pair
from collections import OrderedDict


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = t.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w-m) / t.sqrt(v+1e-5)
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResBackbone(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, stride=1):
        super().__init__()
        mid_ch = mid_ch or in_ch

        self.conv1 = nn.Sequential(
            StdConv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.RReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            StdConv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch)
        )
        self.act = nn.RReLU(inplace=True)
        self.conv3 = nn.Sequential(
            StdConv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.RReLU(inplace=True)
        )

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x + res)
        x = self.conv3(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_ch=4, down_ch=[16, 32, 64, 128, 256]):
        super().__init__()
        # 4 -> [32, 64, 128, 256, 512]

        self.head = nn.Sequential(
            StdConv2d(in_ch, down_ch[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(down_ch[0]),
            nn.RReLU(inplace=True)
        )
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(
                ResBackbone(down_ch[0], down_ch[1]),
                nn.MaxPool2d(2, 2, 0)
            )),
            ('block2', nn.Sequential(
                ResBackbone(down_ch[1], down_ch[2]),
                nn.MaxPool2d(2, 2, 0)
            )),
            ('block3', nn.Sequential(
                ResBackbone(down_ch[2], down_ch[3]),
                nn.MaxPool2d(2, 2, 0)
            )),
            ('block4', nn.Sequential(
                ResBackbone(down_ch[3], down_ch[4]),
                nn.MaxPool2d(2, 2, 0)
            )),
        ]))

    def forward(self, x):
        features = []
        x = self.head(x)
        features.append(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            features.append(x)
        x = self.body[-1](x)
        return x, features[::-1]


class Attention(nn.Module):
    """
        :params `config.hidden_size`: 512
    """
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_att_heads = config.transformer__num_heads
        self.att_head_size = int(config.hidden_size / self.num_att_heads)
        self.all_head_size = self.num_att_heads * self.att_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)

        self.softmax = nn.Softmax(dim=-1)
    
    def reshape_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_att_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.reshape_for_scores(q)
        k = self.reshape_for_scores(k)
        v = self.reshape_for_scores(v)

        qk = t.matmul(q, k.transpose(-1, -2))
        qk = qk / math.sqrt(self.att_head_size)
        qk = self.softmax(qk)

        qkv = t.matmul(qk, v)
        qkv = qkv.permute(0, 2, 1, 3).contiguous()
        new_qkv_shape = qkv.size()[:-2] + (self.all_head_size,)
        qkv = qkv.view(*new_qkv_shape)

        att_output = self.out(qkv)

        return att_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer__mlp_dim)
        self.fc2 = nn.Linear(config.transformer__mlp_dim, config.hidden_size)
        self.act = nn.functional.gelu
        self.dropout = nn.Dropout(config.transformer__dropout_rate)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embedding(nn.Module):
    def __init__(self, config, img_size):
        super(Embedding, self).__init__()
        self.config = config

        self.patch_embedding = nn.Conv2d(256, config.hidden_size, kernel_size=1, stride=1)
        self.pos_embedding = nn.Parameter(t.zeros(1, 144, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer__dropout_rate)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)


        embedding = x + self.pos_embedding
        embedding = self.dropout(embedding)
        return embedding


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention = Attention(config)
        self.mlp = Mlp(config)
        self.att_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, x):
        res = x
        x = self.att_norm(x)
        x = self.attention(x)
        x = x + res

        res = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + res
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.RReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.RReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.RReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        x = t.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Decoders(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.hidden_size),
            nn.RReLU(inplace=True)
        )
        in_chs = [256, 128, 64, 32]
        out_chs = [128, 64, 32, 16]
        layers = [
            Decoder(in_ch, out_ch) for in_ch, out_ch in zip(in_chs, out_chs)
        ]
        self.decoders = nn.ModuleList(layers)
        self.out = nn.Sequential(
            nn.Conv2d(out_chs[-1], 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.RReLU(inplace=True)
        )
    
    def forward(self, x, features):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv(x)
        for i, layer in enumerate(self.decoders):
            x = layer(x, features[i])
        x = self.out(x)
        return x


class TransResNet(nn.Module):
    def __init__(self, config, img_size=[192, 192]):
        super(TransResNet, self).__init__()
        self.resnet = ResNet()
        self.embedding = Embedding(config, img_size=img_size)
        
        self.transformers = nn.ModuleList()
        self.transformers_norm = nn.LayerNorm(config.hidden_size)
        for _ in range(config.transformer__num_layers):
            layer = Transformer(config)
            self.transformers.append(layer)
        self.decoders = Decoders(config)

    def forward(self, x):
        x, features = self.resnet(x)
        x = self.embedding(x) # torch.Size([2, 144, 512])
        for transformer in self.transformers:
            x = transformer(x)
        x = self.transformers_norm(x)
        x = self.decoders(x, features)

        return x
