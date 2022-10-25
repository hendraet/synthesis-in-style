from collections import OrderedDict
from typing import Union, Type

import torch
import torch.nn.functional as F
from torch import nn

from networks.base_segmenter import BaseSegmenter


class DocUFCN(BaseSegmenter):

    def __init__(self, num_classes: int, input_channels: int = 3, encoder_dropout_prob: float = 0.4,
                 decoder_dropout_prob: float = 0.4, background_class_id: int = 0, min_confidence: float = 0.7,
                 min_contour_area: int = 55):
        super().__init__(background_class_id, min_confidence, min_contour_area)
        self.num_classes = num_classes
        self.num_input_channels = input_channels
        self.encoder_dropout_prob = encoder_dropout_prob
        self.decoder_dropout_prob = decoder_dropout_prob
        self.min_contour_area = min_contour_area

        self.feature_sizes = [32, 64, 128, 256]
        self.encoder_blocks = self.build_encoder(input_channels)
        self.decoder_blocks = self.build_decoder()
        self.classifier = nn.Conv2d(2 * self.feature_sizes[0], num_classes, kernel_size=3, padding=1)

    def build_encoder(self, input_channels: int) -> nn.ModuleList:
        encoder_feature_sizes = [input_channels] + self.feature_sizes
        encoder_blocks = []
        for in_planes, out_planes in zip(encoder_feature_sizes, encoder_feature_sizes[1:]):
            encoder_blocks.append(self.build_encoder_conv_block(in_planes, out_planes))
        return nn.ModuleList(encoder_blocks)

    def build_decoder(self) -> nn.ModuleList:
        feature_sizes = list(reversed(self.feature_sizes))
        decoder_blocks = [self.build_decoder_conv_block(feature_sizes[0], feature_sizes[1])]
        for in_planes, out_planes in zip(feature_sizes[1:], feature_sizes[2:]):
            decoder_blocks.append(self.build_decoder_conv_block(2 * in_planes, out_planes))
        return nn.ModuleList(decoder_blocks)

    def build_conv_layer(self, in_size: int, out_size: int, dropout_prob: float, /, dilation: int = 1, conv_class: Union[Type[nn.Conv2d], Type[nn.ConvTranspose2d]] = nn.Conv2d, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Module:
        layers = {
            "conv": conv_class(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            "bn": nn.BatchNorm2d(out_size),
            "relu": nn.ReLU(),
            "dropout": nn.Dropout(dropout_prob)
        }
        return nn.Sequential(OrderedDict(layers))

    def calc_padding(self, in_size, out_size, kernel_size, stride, dilation):
        return int(-(in_size - kernel_size - (kernel_size - 1) * (dilation - 1) - (out_size - 1) * stride) / 2)

    def build_encoder_conv_block(self, in_planes: int, out_planes: int) -> nn.Module:
        conv_layers = [self.build_conv_layer(in_planes, out_planes, self.encoder_dropout_prob, dilation=1)]
        for dilation_factor in [2, 4, 8, 16]:
            padding = self.calc_padding(out_planes, out_planes, 3, 1, dilation_factor)
            conv_layers.append(self.build_conv_layer(out_planes, out_planes, self.encoder_dropout_prob, dilation=dilation_factor, padding=padding))
        return nn.Sequential(*conv_layers)

    def build_decoder_conv_block(self, in_planes: int, out_planes: int) -> nn.Module:
        layers = {
            "conv": self.build_conv_layer(in_planes, out_planes, self.decoder_dropout_prob),
            "upsample": self.build_conv_layer(out_planes, out_planes, self.decoder_dropout_prob, kernel_size=2, stride=2, padding=0, conv_class=nn.ConvTranspose2d)
        }
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_results = []
        h = self.encoder_blocks[0](x)
        for encoder_block in self.encoder_blocks[1:]:
            block_results.append(h.clone())
            h = F.max_pool2d(h, 2, stride=2)
            h = encoder_block(h)

        for decoder_block, encoder_result in zip(self.decoder_blocks, reversed(block_results)):
            h = decoder_block(h)
            h = torch.cat([h, encoder_result], dim=1)

        return self.classifier(h)


class DocUFCNNoDropout(DocUFCN):

    def build_conv_layer(self, in_size: int, out_size: int, dropout_prob: float, /, dilation: int = 1, conv_class: Union[Type[nn.Conv2d], Type[nn.ConvTranspose2d]] = nn.Conv2d, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Module:
        layers = {
            "conv": conv_class(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            "bn": nn.BatchNorm2d(out_size),
            "relu": nn.ReLU()
        }
        return nn.Sequential(OrderedDict(layers))


class PixelShuffleDocUFCN(DocUFCN):

    def build_decoder_conv_block(self, in_planes: int, out_planes: int) -> nn.Module:
        layers = {
            "conv": self.build_conv_layer(in_planes, out_planes * 4, self.decoder_dropout_prob),
            "upsample": nn.PixelShuffle(2)
        }
        return nn.Sequential(OrderedDict(layers))
