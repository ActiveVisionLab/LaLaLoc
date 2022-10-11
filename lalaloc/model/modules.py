import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .position_encoding import build_position_encoding
from .transformer import TransformerEncoder, TransformerEncoderLayer


class EmbeddingModule(nn.Module):
    def __init__(self, in_channels, desc_channels):
        super(EmbeddingModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, desc_channels)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MLPEmbeddingModule(nn.Module):
    def __init__(self, in_channels, desc_channels):
        super(MLPEmbeddingModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, desc_channels),
        )

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


class TransformerFCEmbeddingModule(nn.Module):
    def __init__(
        self,
        in_channels,
        desc_channels,
        pos_at_input=True,
        hidden_dim=2048,
        num_heads=8,
        num_blocks=2,
    ):
        super().__init__()
        self.position_encoder = build_position_encoding(hidden_dim=desc_channels)
        self.dim_reduction = nn.Conv2d(in_channels, desc_channels, 1)
        encoder_layer = TransformerEncoderLayer(desc_channels, num_heads, hidden_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(desc_channels, desc_channels)
        self.pos_at_input = pos_at_input

    def forward(self, x):
        pos = self.position_encoder(x)
        x = self.dim_reduction(x)
        b, c, h, w = x.shape

        x = x.flatten(2).permute(2, 0, 1)  # NxCxHxW -> HWxNxC
        pos = pos.flatten(2).permute(2, 0, 1)  # NxCxHxW -> HWxNxC
        if self.pos_at_input:
            x = x + pos
            pos = None

        x = self.encoder(x, pos=pos).permute(1, 2, 0).view(b, c, h, w)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class PanoModule(nn.Module):
    def __init__(self, config):
        super(PanoModule, self).__init__()
        desc_length = config.MODEL.DESC_LENGTH
        normalise = config.MODEL.NORMALISE_EMBEDDING
        pos_at_input = config.MODEL.PANORAMA_MODULE.POS_AT_INPUT
        num_blocks = config.MODEL.PANORAMA_MODULE.NUM_BLOCKS
        hidden_dim = config.MODEL.PANORAMA_MODULE.HIDDEN_DIM
        net, out_dim = _create_backbone(config.MODEL.PANORAMA_BACKBONE)
        self.layers = nn.Sequential(*list(net.children())[:-2])
        if config.MODEL.PANO_EMBEDDER_TYPE == "fc":
            self.embedding = EmbeddingModule(out_dim, desc_length)
        elif config.MODEL.PANO_EMBEDDER_TYPE == "mlp":
            self.embedding = MLPEmbeddingModule(out_dim, desc_length)
        elif config.MODEL.PANO_EMBEDDER_TYPE == "transformer-fc":
            self.embedding = TransformerFCEmbeddingModule(
                out_dim,
                desc_length,
                pos_at_input=pos_at_input,
                num_blocks=num_blocks,
                hidden_dim=hidden_dim,
            )
        self.normalise = normalise

    def forward(self, x):
        x = self.layers(x)
        x = self.embedding(x)
        if self.normalise:
            x = F.normalize(x)
        return x


class LayoutModule(PanoModule):
    def __init__(self, config):
        super(LayoutModule, self).__init__(config)
        desc_length = config.MODEL.DESC_LENGTH
        net, out_dim = _create_backbone(config.MODEL.LAYOUT_BACKBONE)
        layers = [
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        ]
        layers += list(net.children())[1:-2]
        self.layers = nn.Sequential(*layers)
        if config.MODEL.LAYOUT_EMBEDDER_TYPE == "fc":
            self.embedding = EmbeddingModule(out_dim, desc_length)
        elif config.MODEL.LAYOUT_EMBEDDER_TYPE == "mlp":
            self.embedding = MLPEmbeddingModule(out_dim, desc_length)


class LayoutDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        desc_length = config.MODEL.DESC_LENGTH
        self.fc = nn.Sequential(nn.Linear(desc_length, 2048), nn.ReLU())
        upsample_layers = [
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 1),
        ]
        self.decov_layers = nn.Sequential(*upsample_layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 2, 4)
        x = self.decov_layers(x)
        return x


def _create_backbone(name):
    backbones = {
        "resnet18": (models.resnet18(pretrained=True), 512),
        "resnet50": (models.resnet50(pretrained=True), 2048),
    }
    return backbones[name]
