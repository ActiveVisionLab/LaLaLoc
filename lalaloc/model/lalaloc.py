from logging import warn
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lalaloc_base import Image2LayoutBase, Layout2LayoutBase
from .modules import LayoutDecoder
from .losses import triplet_loss, bbs_loss


class ImageFromLayout(Image2LayoutBase):
    def __init__(self, config):
        super(ImageFromLayout, self).__init__(config)
        self.load_weights_from_l2l(config.TRAIN.SOURCE_WEIGHTS)

    def load_weights_from_l2l(self, ckpt_path):
        if not ckpt_path:
            warnings.warn("No source for the layout branch weights was specified")
            return
        # load weights from Layout2Layout model
        ckpt_dict = torch.load(ckpt_path)
        model_weights = ckpt_dict["state_dict"]

        # load "embedder" weights into "reference_embedder"
        load_dict = {}
        for k, v in model_weights.items():
            modules = k.split(".")
            parent = modules[0]
            if parent == "embedder":
                child = ".".join(modules[1:])
                load_dict[child] = v
        self.reference_embedder.load_state_dict(load_dict)

        # freeze reference_embedder weights
        for p in self.reference_embedder.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        for m in self.reference_embedder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        # compute query and reference embeddings
        query_image = batch["panorama"]
        query_embed = self.forward(q=query_image)

        reference_layouts = batch["pano_layout"].unsqueeze(1)
        reference_embed = self.forward(r=reference_layouts).squeeze(1)

        # perform L2 distance loss
        loss = ((query_embed - reference_embed) ** 2).sum(dim=1).sqrt().mean()

        stats_to_log = {"train/l2_loss": loss.item()}
        return {"loss": loss, "log": stats_to_log}


class Layout2LayoutDecode(Layout2LayoutBase):
    def __init__(self, config):
        super(Layout2LayoutDecode, self).__init__(config)
        self.layout_decoder = LayoutDecoder(config)

    def training_step(self, batch, batch_idx):
        query_image = batch[self.query_key]
        reference_layouts = batch["sampled_layouts"]
        gt_distances = batch["distances"]

        query_embed = self.forward(q=query_image)
        reference_embed = self.forward(r=reference_layouts)

        # perform layout2layout layout loss
        distances = self.compute_distances(query_embed, reference_embed)
        if self.config.TRAIN.LOSS == "triplet":
            loss_layout = triplet_loss(distances)
        elif self.config.TRAIN.LOSS == "bbs":
            gt_distances = gt_distances.float().to(self.device)
            loss_layout = bbs_loss(distances, gt_distances)
        else:
            raise NotImplementedError(
                "{} loss type is not currently implemented".format(
                    self.config.TRAIN.LOSS
                )
            )

        # decode the layout embedding and compute loss
        query_decoded = self.layout_decoder(query_embed)
        query_target = F.interpolate(
            query_image.detach().clone(), self.config.MODEL.DECODER_RESOLUTION
        )
        reference_decoded = self.layout_decoder(reference_embed)

        h, w = reference_layouts.shape[-2:]
        reference_targets = F.interpolate(
            reference_layouts.view(-1, 1, h, w).detach().clone(),
            self.config.MODEL.DECODER_RESOLUTION,
        )

        decoded = torch.cat([query_decoded, reference_decoded], dim=0)
        target = torch.cat([query_target, reference_targets], dim=0)
        loss_decode = F.l1_loss(decoded, target)

        loss = (
            self.config.TRAIN.DECODER_LOSS_SCALE * loss_decode
            + self.config.TRAIN.LAYOUT_LOSS_SCALE * loss_layout
        )
        stats_to_log = {
            "train/loss": loss.item(),
            "train/layout_loss": loss_layout.item(),
            "train/decoder_loss": loss_decode.item(),
        }
        return {"loss": loss, "log": stats_to_log}
