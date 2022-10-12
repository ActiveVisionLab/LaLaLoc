import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=False
        )
        self.conv = ConvBlock(in_channels * 2, out_channels)

    def forward(self, x, encoder_x):
        x = self.upsample(x)
        x = torch.cat([x, encoder_x], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        in_channels = 3
        blocks = []
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return x, features


class Decoder(nn.Module):
    def __init__(self, channels, in_channels):
        super().__init__()
        blocks = []
        self.initial_block = DecodeBlock(in_channels, channels[0])
        for out_channels in channels[1:]:
            blocks.append(DecodeBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, encoder_features):
        for block, encoder_x in zip(self.blocks, encoder_features):
            x = block(x, encoder_x)
        return x


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_channels = config.MODEL.UNET_ENCODER_CHANNELS
        decoder_channels = config.MODEL.UNET_DECODER_CHANNELS
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels, encoder_channels[-1])
        self.conv = nn.Conv2d(128, 128, 3, padding=1)

    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features[::-1])
        x = self.conv(x)
        x = F.normalize(x, dim=1)
        return x
