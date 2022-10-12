import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from ..utils.floorplan import (
    create_pixel_loc_grid,
    pixel_loc_to_pose,
    pose_to_pixel_loc,
    sample_locs,
)
from .lalaloc_base import Image2LayoutBase
from .pose_optimisation import PoseConvergenceChecker, init_optimiser
from .unet import UNet


class FloorPlanUnetBase(Image2LayoutBase):
    def __init__(self, config):
        super(FloorPlanUnetBase, self).__init__(config)
        # Remove uneeded LaLaLoc layout branch
        self.reference_embedder = None
        # Create LaLaLoc++ plan branch
        self.floorplan_encoder = UNet(config)

    def predict_pose_dense(
        self, query_desc, plan_embed, plan_scale, plan_shift, query_z, mask=None
    ):
        _, c, h, w = plan_embed.shape
        n = query_desc.shape[0]
        dense_loc_grid = create_pixel_loc_grid(w, h)
        dense_pose_grid = pixel_loc_to_pose(
            dense_loc_grid, plan_scale.cpu(), plan_shift.cpu(), query_z
        ).cpu()
        plan_embed_ = plan_embed.clone()
        if mask is not None:
            dense_pose_grid = dense_pose_grid[mask, :]
            plan_embed_ = plan_embed_[:, :, mask]
        dense_poses = dense_pose_grid.view(-1, 3)
        plan_embed_ = plan_embed_.view(c, -1).transpose(0, 1)

        plan_embed_ = plan_embed_.unsqueeze(0).expand(n, -1, -1)
        pred_distances = self.compute_distances(query_desc, plan_embed_).detach().cpu()
        ranking_dense = torch.argsort(pred_distances, dim=-1)[:, :5]
        pred_poses = dense_poses[ranking_dense[:, 0]].to(self.device)
        return pred_poses

    def optimise_pose(
        self, query_embeddings, initial_poses, feature_map, plan_scale, plan_shift
    ):
        torch.set_grad_enabled(True)
        initial_locs = pose_to_pixel_loc(
            initial_poses.unsqueeze(1), plan_scale, plan_shift
        )
        refined_locs = []
        for query_embedding, initial_loc in zip(query_embeddings, initial_locs):
            offset = torch.zeros((1, 2), requires_grad=True, device=self.device)

            # initialise optimisation and stopping metrics
            optimiser, scheduler = init_optimiser(self.config, [offset])
            convergence_checker = PoseConvergenceChecker(self.config)

            for j in range(self.config.POSE_REFINE.MAX_ITERS):
                optimiser.zero_grad()
                embedding = sample_locs(
                    feature_map, (initial_loc + offset).unsqueeze(0)
                )
                loss = torch.norm(query_embedding - embedding, dim=-1)
                loss.backward()

                current_loss = loss.item()
                current_loc = (initial_loc + offset).clone().detach()
                if convergence_checker.has_converged(current_loss, current_loc):
                    break
                optimiser.step()
                scheduler.step(current_loss)
            refined_loc = convergence_checker.best_pose
            refined_locs.append(refined_loc)
        refined_locs = torch.stack(refined_locs)
        z = initial_poses[0, -1]
        refined_poses = pixel_loc_to_pose(refined_locs, plan_scale, plan_shift, z).view(
            -1, 3
        )
        torch.set_grad_enabled(False)
        return refined_poses
