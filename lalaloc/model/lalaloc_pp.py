import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from ..data.dataset import TargetEmbeddingDataset
from ..utils.floorplan import pose_to_pixel_loc, sample_locs
from .lalaloc_pp_base import FloorPlanUnetBase
from .losses import bbs_loss
from .modules import LayoutDecoder
from .unet import UNet

ROOM_VALUE = 0.5


def visualise_features(features, valid_mask=None):
    features = features.cpu()
    if valid_mask is not None:
        valid_mask = valid_mask.cpu()
    for feature_map in features:
        h, w = feature_map.shape[-2:]
        feature_map = feature_map.view(-1, h * w).transpose(0, 1).numpy()
        pca = PCA(n_components=3)
        feature_map_pca = pca.fit_transform(feature_map)
        feature_map_pca = feature_map_pca.reshape(h, w, 3)

        shift = np.min(feature_map_pca)
        feature_map_pca -= shift
        scale = np.max(feature_map_pca)
        feature_map_pca /= scale

        if valid_mask is not None:
            invalid_mask = ~valid_mask
            feature_map_pca[invalid_mask] = 1.0

        return feature_map_pca


class FloorPlanUnetLayout(FloorPlanUnetBase):
    def __init__(self, config):
        super(FloorPlanUnetLayout, self).__init__(config)
        self.layout_decoder = LayoutDecoder(config)

    def training_step(self, batch, batch_idx):
        plans = batch["floorplan"]
        plan_params = batch["floorplan_params"]
        plan_scale = plan_params["scale"]
        plan_shift = plan_params["shift"]
        plan_heights = plan_params["h"]
        plan_widths = plan_params["w"]
        query_layouts = batch["pano_layout"]
        query_pose = batch["pano_pose"]
        reference_layouts = batch["sampled_layouts"]
        reference_poses = batch["sampled_poses"]

        query_locs = pose_to_pixel_loc(query_pose.unsqueeze(1), plan_scale, plan_shift)
        reference_locs = pose_to_pixel_loc(reference_poses, plan_scale, plan_shift)

        # embed floor plan and sample locations
        query_embed = []
        reference_embed = []
        for plan, query_loc, reference_loc, h, w in zip(
            plans, query_locs, reference_locs, plan_heights, plan_widths
        ):
            plan_embed = self.floorplan_encoder(plan[:, :h, :w].unsqueeze(0))
            qry_embed = sample_locs(
                plan_embed,
                query_loc.unsqueeze(0),
                normalise=self.config.MODEL.NORMALISE_SAMPLE,
            )
            ref_embed = sample_locs(
                plan_embed,
                reference_loc.unsqueeze(0),
                normalise=self.config.MODEL.NORMALISE_SAMPLE,
            )
            query_embed.append(qry_embed)
            reference_embed.append(ref_embed)
        query_embed = torch.cat(query_embed).squeeze(1)
        reference_embed = torch.cat(reference_embed)

        # decode the layout embeddings for both queries and reference
        query_decoded = self.layout_decoder(query_embed)
        query_target = F.interpolate(
            query_layouts.detach().clone(), self.config.MODEL.DECODER_RESOLUTION
        )
        reference_decoded = self.layout_decoder(reference_embed)
        h, w = reference_layouts.shape[-2:]
        reference_targets = F.interpolate(
            reference_layouts.view(-1, 1, h, w).detach().clone(),
            self.config.MODEL.DECODER_RESOLUTION,
        )
        decoded = torch.cat([query_decoded, reference_decoded], dim=0)

        # compute decoding loss
        target = torch.cat([query_target, reference_targets], dim=0)
        loss_decode = F.l1_loss(decoded, target)
        loss = self.config.TRAIN.DECODER_LOSS_SCALE * loss_decode
        stats_to_log = {"train/decoder_loss": loss_decode.item()}

        # compute bbs loss if specified
        if self.config.TRAIN.LOSS == "decoder_plus_bbs":
            gt_distances = batch["distances"]
            gt_distances = gt_distances.float().to(self.device)
            distances = self.compute_distances(query_embed, reference_embed)
            loss_layout = bbs_loss(distances, gt_distances)
            stats_to_log["train/layout_loss"] = loss_layout.item()
            loss = loss + self.config.TRAIN.LAYOUT_LOSS_SCALE * loss_layout

        stats_to_log["train/loss"] = loss.item()
        return {"loss": loss, "log": stats_to_log}

    def inference_step(self, batch, batch_idx):
        plan_params = batch["floorplan_params"]
        plan_height = plan_params["h"]
        plan_width = plan_params["w"]
        plan = batch["floorplan"][:, :plan_height, :plan_width].unsqueeze(0)
        plan_scale = torch.Tensor([plan_params["scale"]]).to(self.device)
        plan_shift = plan_params["shift"]

        plan_embed = self.floorplan_encoder(plan).detach()

        # plot latent floor plan
        valid_loc_mask = plan[0, 0] == ROOM_VALUE
        vis_features = visualise_features(plan_embed, valid_loc_mask)
        self.logger.experiment.add_image(
            f"unet_feats_{batch_idx}",
            vis_features,
            self.current_epoch,
            dataformats="HWC",
        )

        # sample embedding at query location
        query_pose = batch["pano_pose"]
        query_z = query_pose[0, -1]
        query_loc = pose_to_pixel_loc(
            query_pose.unsqueeze(0).clone(), plan_scale, plan_shift
        )
        query_embed = sample_locs(plan_embed, query_loc).squeeze(0)

        # legacy sampling of sparse grid to emulate LaLaLoc
        reference_poses = batch["sampled_poses"]
        reference_locs = pose_to_pixel_loc(
            reference_poses.unsqueeze(0).clone(), plan_scale, plan_shift
        )
        reference_embed = sample_locs(plan_embed, reference_locs).squeeze(0)
        n, _ = query_embed.shape
        reference_embed = reference_embed.unsqueeze(0).expand(n, -1, -1)

        distances = self.compute_distances(query_embed, reference_embed).detach().cpu()
        gt_distances = batch["distances"].cpu()
        pose_distances = torch.norm(
            (query_pose.unsqueeze(1) - reference_poses.expand(n, -1, -1)), dim=-1
        ).cpu()

        # gather ranking info for the prediction vs the actual
        ranking_prediction = torch.argsort(distances, dim=-1)[:, :5]
        ranking_layout = torch.argsort(gt_distances, dim=-1)[:, :5]
        ranking_pose = torch.argsort(pose_distances, dim=-1)[:, :5]

        retrieval_error = pose_distances.gather(
            1, ranking_prediction[:, 0].unsqueeze(1)
        )
        oracle_error = pose_distances.gather(1, ranking_layout[:, 0].unsqueeze(1))

        # retrieve from dense LaLaLoc++ prediction
        refined_poses = optimised_poses = self.predict_pose_dense(
            query_embed, plan_embed, plan_scale, plan_shift, query_z, valid_loc_mask
        ).to(self.device)

        refined_error = torch.norm(query_pose - refined_poses, dim=-1, keepdim=True)
        optimised_error = torch.norm(query_pose - optimised_poses, dim=-1, keepdim=True)

        return {
            "pred_rank": ranking_prediction,
            "layout_rank": ranking_layout,
            "pose_rank": ranking_pose,
            "retrieval_error": retrieval_error,
            "optimised_error": optimised_error,
            "refined_error": refined_error,
            "oracle_error": oracle_error,
            "pred_pose": optimised_poses,
        }


class FloorPlanUnetImage(FloorPlanUnetBase):
    def __init__(self, config):
        super(FloorPlanUnetImage, self).__init__(config)
        self.load_weights_from_plan_only(config.TRAIN.SOURCE_WEIGHTS)
        for p in self.floorplan_encoder.parameters():
            p.requires_grad = False

    def load_weights_from_plan_only(self, ckpt_path):
        if not ckpt_path:
            warnings.warn("No source for the layout branch weights was specified")
            return
        print("Loading Floor Plan Encoder from {}".format(ckpt_path))
        # load weights from plan-branch-only model
        ckpt_dict = torch.load(ckpt_path)
        model_weights = ckpt_dict["state_dict"]

        # load "embedder" weights into "reference_embedder"
        load_dict = {}
        for k, v in model_weights.items():
            modules = k.split(".")
            parent = modules[0]
            if parent == "floorplan_encoder":
                child = ".".join(modules[1:])
                load_dict[child] = v
        self.floorplan_encoder.load_state_dict(load_dict)

    def train_dataloader(self):
        dataset = TargetEmbeddingDataset(
            self.floorplan_encoder, self.config, device=self.device
        )
        batch_size = self.config.TRAIN.BATCH_SIZE
        num_workers = self.config.SYSTEM.NUM_WORKERS
        dataloader = DataLoader(
            dataset, batch_size, shuffle=True, num_workers=num_workers,
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        query_image = batch["panorama"]
        target_embed = batch["target_embedding"]

        query_embed = self.forward(q=query_image)

        loss = ((target_embed - query_embed) ** 2).sum(dim=1).sqrt().mean()
        stats_to_log = {"train/loss": loss.item()}
        return {"loss": loss, "log": stats_to_log}

    def inference_step(self, batch, batch_idx):
        plan_params = batch["floorplan_params"]
        plan_height = plan_params["h"]
        plan_width = plan_params["w"]
        plan = batch["floorplan"][:, :plan_height, :plan_width].unsqueeze(0)
        plan_scale = torch.Tensor([plan_params["scale"]]).to(self.device)
        plan_shift = plan_params["shift"]

        plan_embed = self.floorplan_encoder(plan).detach()

        # if specified subsample the embedded floor plan
        subsample_x = self.config.TEST.SUBSAMPLE_PLAN_X
        if subsample_x > 1:
            plan_embed = plan_embed[:, :, ::subsample_x, ::subsample_x]
            plan = plan[:, :, ::subsample_x, ::subsample_x]
            plan_scale = plan_scale / subsample_x

        # sample embedding at query location
        query_image = batch[self.query_key]
        query_pose = batch["pano_pose"]
        query_embed = self.forward(q=query_image).detach()

        # plot latent floor plan
        query_z = query_pose[0, -1]
        valid_loc_mask = plan[0, 0] == ROOM_VALUE
        vis_features = visualise_features(plan_embed, valid_loc_mask)
        self.logger.experiment.add_image(
            f"unet_feats_{batch_idx}",
            vis_features,
            self.current_epoch,
            dataformats="HWC",
        )

        # legacy sampling of sparse grid to emulate LaLaLoc
        reference_poses = batch["sampled_poses"]
        reference_locs = pose_to_pixel_loc(
            reference_poses.unsqueeze(0).clone(), plan_scale, plan_shift
        )
        reference_embed = sample_locs(plan_embed, reference_locs).squeeze(0)
        n, _ = query_embed.shape
        reference_embed = reference_embed.unsqueeze(0).expand(n, -1, -1)

        distances = self.compute_distances(query_embed, reference_embed).detach().cpu()
        pose_distances = torch.norm(
            (query_pose.unsqueeze(1) - reference_poses.expand(n, -1, -1)), dim=-1
        ).cpu()
        # gather ranking info for the prediction vs the actual
        ranking_prediction = torch.argsort(distances, dim=-1)[:, :5]
        ranking_pose = torch.argsort(pose_distances, dim=-1)[:, :5]

        retrieval_error = pose_distances.gather(
            1, ranking_prediction[:, 0].unsqueeze(1)
        )

        if self.config.TEST.COMPUTE_GT_DIST:
            gt_distances = batch["distances"].cpu()
            ranking_layout = torch.argsort(gt_distances, dim=-1)[:, :5]
            oracle_error = pose_distances.gather(1, ranking_layout[:, 0].unsqueeze(1))
        else:
            oracle_error = torch.zeros_like(retrieval_error)
            ranking_layout = torch.zeros_like(ranking_prediction)

        # retrieve from dense prediction and optimise them
        refined_poses = self.predict_pose_dense(
            query_embed, plan_embed, plan_scale, plan_shift, query_z, valid_loc_mask
        )
        optimised_poses = self.optimise_pose(
            query_embed, refined_poses.clone(), plan_embed, plan_scale, plan_shift
        )

        refined_error = torch.norm(query_pose - refined_poses, dim=-1, keepdim=True)
        optimised_error = torch.norm(query_pose - optimised_poses, dim=-1, keepdim=True)
        return {
            "pred_rank": ranking_prediction,
            "layout_rank": ranking_layout,
            "pose_rank": ranking_pose,
            "retrieval_error": retrieval_error,
            "optimised_error": optimised_error,
            "refined_error": refined_error,
            "oracle_error": oracle_error,
            "pred_pose": optimised_poses,
        }

