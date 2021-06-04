import torch as t
import torch.nn as nn

from torch.autograd import Variable

import numpy as np
import warnings
import math
import copy

from torch.nn.modules.utils import _pair
from collections import OrderedDict


class SingleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_ch, out_ch),
            # nn.BatchNorm3d(out_ch),
            nn.RReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_ch, out_ch),
            # nn.BatchNorm3d(out_ch),
            nn.RReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(out_ch, out_ch),
            # nn.BatchNorm3d(out_ch)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResConvBlock, self).__init__()
        self.conv1 = DoubleConvBlock(in_ch, in_ch)
        self.relu = nn.RReLU(inplace=True)
        self.conv2 = SingleConvBlock(in_ch, out_ch)
    
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.relu(x + res)
        x = self.conv2(x)
        return x


class ConvUpBlock(nn.Module):
    def __init__(self, in_data, out_data, kernel=3, stride=1, padding=1):
        super(ConvUpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_data, out_data, kernel_size=3, stride=2, padding=1, output_padding=1,
                               dilation=1),
            nn.GroupNorm(out_data, out_data),
            # nn.BatchNorm3d(out_data),
            nn.RReLU(inplace=True)
        )
        self.down = nn.Sequential(
            DoubleConvBlock(2*out_data, out_data),
            nn.RReLU(inplace=True)
        )
    
    def forward(self, x, down_features):
        x = self.up(x)
        x = t.cat([x, down_features], dim=1)
        x = self.down(x)
        return x


class Attention(nn.Module):
    """
        :params `config.hidden_size`: 256
    """
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_att_heads = config.transformer__num_heads # 16
        self.att_head_size = int(config.hidden_size / self.num_att_heads) # 256 / 16 = 16
        self.all_head_size = self.num_att_heads * self.att_head_size # 16 * 16 = 256

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


'''
input.shape: torch.Size([1, 4, 192, 192, 48])
resnet.shape: torch.Size([1, 256, 12, 12, 3])
embedding.shape: torch.Size([1, 432, 256])
output.shape: torch.Size([1, 4, 192, 192, 48])
'''

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config

        max_len = 256
        pe = t.zeros(max_len, config.hidden_size)
        position = t.arange(0., max_len).unsqueeze(1)
        div_term = t.exp(t.arange(0., config.hidden_size, 2) * (-math.log(10000.0)/config.hidden_size))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(config.transformer__dropout_rate)

    def forward(self, x):
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


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


class ResNet(nn.Module):
    def __init__(self, in_ch):
        super(ResNet, self).__init__()
        in_kn = [16, 32, 64, 128]
        out_kn = [32, 64, 128, 256]
        self.in_model = nn.Sequential(
            DoubleConvBlock(in_ch, 16),
            nn.RReLU(inplace=True)
        )
        self.body = nn.Sequential(OrderedDict([
            (f'resblock_{i:d}', nn.Sequential(
                ResConvBlock(i_kn, o_kn),
                nn.MaxPool3d(2, 2, 0)
            )) for i, (i_kn, o_kn) in enumerate(zip(in_kn, out_kn))
        ]))

    def forward(self, x):
        features = []
        x = self.in_model(x)
        features.append(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            features.append(x)
        x = self.body[-1](x)
        return x, features[::-1]


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1,
                               dilation=1),
            nn.GroupNorm(out_ch, out_ch),
            # nn.BatchNorm3d(out_ch),
            nn.RReLU(inplace=True),
        )
        self.down = nn.Sequential(
            DoubleConvBlock(2*out_ch, out_ch),
            nn.RReLU(inplace=True),
        )
    
    def forward(self, x, down_features):
        x = self.up(x)
        x = t.cat([x, down_features], dim=1)
        x = self.down(x)
        return x


class Decoders(nn.Module):
    def __init__(self, config, trans=True):
        super().__init__()
        self.trans = trans
        self.conv = nn.Sequential(
            DoubleConvBlock(config.hidden_size, config.hidden_size),
            nn.RReLU(inplace=True)
        )
        # self.conv = SingleConvBlock(config.hidden_size, config.hidden_size)
        in_chs = [256, 128, 64, 32]
        out_chs = [128, 64, 32, 16]
        layers = [
            Decoder(in_ch, out_ch) for in_ch, out_ch in zip(in_chs, out_chs)
        ]
        self.decoders = nn.ModuleList(layers)
        self.out = nn.Sequential(
            DoubleConvBlock(out_chs[-1], 4),
            nn.ReLU(inplace=True)
        )
        # self.out = SingleConvBlock(out_chs[-1], 4)
    
    def forward(self, x, features):
        if self.trans:
            B, N, hidden = x.size()
            k = int(math.pow(N/9, 1/3))
            x = x.permute(0, 2, 1)
            x = x.contiguous().view(B, hidden, int(k*4), int(k*4), k)
        x = self.conv(x)
        for i, layer in enumerate(self.decoders):
            x = layer(x, features[i])
        x = self.out(x)
        return x


"""
⬆: Basic Module

⬇: Main Module
    + 3dresnet
    + embedding
    + transformers
    + decoders
"""


class TransResNet_3d(nn.Module):
    def __init__(self, config):
        super(TransResNet_3d, self).__init__()
        self.resnet = ResNet(in_ch=4)
        self.embedding = Embedding(config)
        self.transformers = nn.ModuleList()
        self.transformers_norm = nn.LayerNorm(config.hidden_size)
        for _ in range(config.transformer__num_layers):
            layer = Transformer(config)
            self.transformers.append(layer)
        self.decoders = Decoders(config)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x, features = self.resnet(x)
        x = self.embedding(x)
        for transformer in self.transformers:
            x = transformer(x)
        x = self.decoders(x, features)
        return x


class ResNet_3d(nn.Module):
    def __init__(self, config):
        super(ResNet_3d, self).__init__()
        self.resnet = ResNet(in_ch=4)
        self.embedding = Embedding(config)
        self.transformers = nn.ModuleList()
        self.transformers_norm = nn.LayerNorm(config.hidden_size)
        for _ in range(config.transformer__num_layers):
            layer = Transformer(config)
            self.transformers.append(layer)
        self.decoders = Decoders(config, trans=False)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x, features = self.resnet(x)
        # x = self.embedding(x)
        # for transformer in self.transformers:
        #     x = transformer(x)
        x = self.decoders(x, features)
        return x


"""notes
192 = 4 * 48 -> 48/16 = 3 for 4 maxpooling 2^4=16
144 = 3 * 48
"""
