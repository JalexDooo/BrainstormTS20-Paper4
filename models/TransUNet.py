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


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = StdConv2d(cin, cmid, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = StdConv2d(cmid, cmid, kernel_size=3, stride=stride, padding=1, bias=False, groups=1)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = StdConv2d(cmid, cout, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            self.downsample = StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)
    
    def forward(self, x):
        res = x
        if hasattr(self, 'downsample'):
            res = self.downsample(x)
            res = self.gn_proj(res)
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = self.relu(res+y)

        return y


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor) # 64
        self.width = width

        # conv, gn, relu
        self.root = nn.Sequential(
            StdConv2d(4, width, kernel_size=7, stride=2, bias=False, padding=3),
            nn.GroupNorm(32, width, eps=1e-6),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))
    
    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size/4/(i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = t.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_att_heads = config.transformer__num_heads # 12
        self.att_head_size = int(config.hidden_size / self.num_att_heads) # 768 // 12 = 64
        self.all_head_size = self.num_att_heads * self.att_head_size # 64 * 12 = 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)

        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_att_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x):
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        att_scores = t.matmul(query_layer, key_layer.transpose(-1, -2))
        att_scores = att_scores / math.sqrt(self.att_head_size)
        att_probs = self.softmax(att_scores)

        context_layer = t.matmul(att_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        att_output = self.out(context_layer)

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
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=4):
        """parameter
        config: config object.
        img_size: [w, h] 256 x 256
        """
        super(Embeddings, self).__init__()
        self.config = config

        grid_size = config.resnet__grid
        patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1]) # 1 x 1
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16) # 16 x 16
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) # 256

        self.res_model = ResNetV2(block_units=config.resnet__num_layers, width_factor=config.resnet__width_factor)
        in_channels = self.res_model.width * 16 # 64 * 16 = 1024

        self.patch_embeddings = nn.Conv2d(in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(t.zeros(1, n_patches, config.hidden_size)) # [1, 256, 768]

        self.dropout = nn.Dropout(config.transformer__dropout_rate)

    def forward(self, x):
        x, features = self.res_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        # print('x.shape: {}, position_embeddings.shape: {}'.format(x.shape, self.position_embeddings.shape))

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.att_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)
    
    def forward(self, x):
        h = x
        x = self.att_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer__num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        
    def forward(self, x):
        for layer_block in self.layer:
            x = layer_block(x)
        x = self.encoder_norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)
    
    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded, features


"""
⬆: Encoder & Transformer
⬇: Decoder & Segmentation
"""


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels+skip_channels, out_channels, 3, 1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, 1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = t.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, 3, 1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        skip_channels = self.config.skip_channels
        for i in range(4-self.config.n_skip):
            skip_channels[3-i] = 0
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, features):
        B, n_patch, hidden = x.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if (i < self.config.n_skip):
                skip = features[i] 
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling>1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class TransUNet(nn.Module):
    def __init__(self, config, img_size=[256, 256]):
        super(TransUNet, self).__init__()
        self.transformer = Transformer(config, img_size)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(config.decoder_channels[-1], config.n_classes, kernel_size=3)
        self.config = config
    
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 4, 1, 1)
        x, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits
