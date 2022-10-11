import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lalaloc.utils.render import render_scene_batched
from lalaloc.utils.vogel_disc import sample_vogel_disc

from ..data.dataset import Structured3DPlans
from ..data.transform import build_transform
from ..utils.eval import recall_at_n
from ..utils.projection import projects_onto_floor
from .modules import LayoutModule, PanoModule
from .pose_optimisation import (
    PoseConvergenceChecker,
    init_camera_at_origin,
    init_objects_at_pose,
    init_optimiser,
    render_at_pose,
)


def build_dataloader(config, split):
    is_train = split == "train"

    batch_size = config.TRAIN.BATCH_SIZE if is_train else None
    num_workers = (
        config.SYSTEM.NUM_WORKERS if is_train or not config.TEST.COMPUTE_GT_DIST else 0
    )

    dataset = Structured3DPlans(config, split)
    dataloader = DataLoader(
        dataset, batch_size, shuffle=is_train, num_workers=num_workers,
    )
    return dataloader


class Image2LayoutBase(pl.LightningModule):
    def __init__(self, config):
        super(Image2LayoutBase, self).__init__()
        self.query_embedder = PanoModule(config)
        self.reference_embedder = LayoutModule(config)
        self.desc_length = config.MODEL.DESC_LENGTH
        self.config = config
        # The key to access the query data type from the batch dict
        self.query_key = "panorama"

    def forward(self, q=None, r=None):
        if q is None and r is None:
            raise Exception

        if q is not None:
            q = self.query_embedder(q)
            if r is None:
                return q

        if r is not None:
            n, m, c, h, w = r.shape
            r = r.reshape(n * m, c, h, w)
            r = self.reference_embedder(r)
            r = r.reshape(n, m, self.desc_length)
            if q is None:
                return r

        d = self.compute_distances(q, r)
        return d

    def compute_distances(self, p, l):
        n, m, _ = l.shape
        p = p.unsqueeze(1).expand(-1, m, -1)
        p = p.reshape(n * m, self.desc_length)
        l = l.reshape(-1, self.desc_length)

        d = F.pairwise_distance(p, l)
        d = d.reshape(n, m)
        return d

    def vogel_refinement(self, query_embeddings, nn_poses, geometry, floor):
        transform = build_transform(self.config, False, is_layout=True)
        radius = 2 * self.config.TEST.POSE_SAMPLE_STEP
        num_samples = self.config.TEST.VOGEL_SAMPLES
        refined_poses = []
        for query_embedding, nn_pose in zip(query_embeddings, nn_poses):
            sampled_poses = sample_vogel_disc(nn_pose, radius, num_samples)
            poses_to_render = []
            for pose in sampled_poses:
                room_idx = projects_onto_floor(pose, floor)
                if room_idx < 0:
                    continue
                poses_to_render.append((pose, room_idx))
            local_layouts = render_scene_batched(self.config, geometry, poses_to_render)
            local_poses = [torch.tensor(p[0]) for p in poses_to_render]

            # transform and stack layouts
            local_layouts = [transform(l) for l in local_layouts]
            local_layouts = torch.stack(local_layouts).to(self.device)
            # feed into embedder
            local_embeddings = self.forward(r=local_layouts.unsqueeze(0)).detach()
            # take min distance between result and query_embedding
            # and append its respective pose
            distances = torch.norm(local_embeddings - query_embedding, dim=-1)
            refined_pose = local_poses[distances.argmin()]
            refined_poses.append(refined_pose)
        refined_poses = torch.stack(refined_poses)
        return refined_poses

    def latent_pose_optimisation(self, query_embeddings, nn_poses, geometry, floor):
        # Ensure gradients are enabled and modules are in eval mode
        torch.set_grad_enabled(True)
        self.eval()

        refined_poses = []
        for query_embedding, nn_pose in zip(query_embeddings, nn_poses):
            query_embedding = query_embedding.detach()
            query_embedding.requires_grad = False

            # gather room geometry
            room_idx = projects_onto_floor(nn_pose, floor)
            mesh = geometry[room_idx]

            # centre geometry at pose
            nn_pose = nn_pose.to(self.device)
            objects, vertices = init_objects_at_pose(nn_pose, mesh, self.device)
            camera = init_camera_at_origin(self.config)

            # initialise the refinement translation vector
            # note: these represent displacements from the initial pose in metres
            pose_xy = torch.zeros((1, 2), requires_grad=True, device=self.device)
            pose_z = torch.zeros((1, 1), requires_grad=False, device=self.device)

            # initialise optimisation and stopping metrics
            optimiser, scheduler = init_optimiser(self.config, [pose_xy])
            convergence_checker = PoseConvergenceChecker(self.config)

            for j in range(self.config.POSE_REFINE.MAX_ITERS):
                optimiser.zero_grad()
                layout = render_at_pose(
                    camera, objects, vertices, torch.cat([pose_xy, pose_z], dim=1)
                )

                h, w = layout.shape
                layout = layout.view(1, 1, h, w)
                # for faster rendering, sometimes we render the layouts at a smaller resolution than the network takes as input
                # therefore, we need to interpolate the layout to the target input size
                if self.config.POSE_REFINE.RENDER_SIZE != self.config.INPUT.IMG_SIZE:
                    layout = F.interpolate(layout, self.config.INPUT.IMG_SIZE)

                layout_embedding = self.forward(r=layout.unsqueeze(0))
                loss = torch.norm(query_embedding - layout_embedding, dim=-1)
                loss.backward()

                # check convergence
                current_loss = loss.item()
                current_pose = torch.cat([pose_xy, pose_z], dim=1).clone().detach()
                if convergence_checker.has_converged(current_loss, current_pose):
                    break

                optimiser.step()
                scheduler.step(current_loss)
            # pose displacement is optimised in metres, therefore convert it to mm
            refined_pose = nn_pose + convergence_checker.best_pose * 1000
            refined_poses.append(refined_pose)
        refined_poses = torch.stack(refined_poses)
        refined_poses = refined_poses.detach().cpu().squeeze(1)
        torch.set_grad_enabled(False)
        return refined_poses

    def configure_optimizers(self):
        optimiser = optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.TRAIN.INITIAL_LR,
            momentum=self.config.TRAIN.MOMENTUM,
            weight_decay=self.config.TRAIN.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimiser, self.config.TRAIN.LR_MILESTONES, self.config.TRAIN.LR_GAMMA
        )
        return [optimiser], [scheduler]

    def training_step(self, batch, batch_idx):
        # Training step should be implemented in child classes
        raise NotImplementedError("The training routine is not implemented.")

    def inference_step(self, batch, batch_idx):
        query_image = batch[self.query_key]
        query_pose = batch["pano_pose"]
        reference_layouts = batch["sampled_layouts"]
        reference_poses = batch["sampled_poses"]

        gt_distances = batch["distances"]
        query_image = query_image
        query_pose = query_pose.cpu()
        reference_poses = reference_poses.cpu()
        gt_distances = gt_distances.cpu()

        # compute the desciptors for the panoramas in each room
        query_desc = self.forward(q=query_image).detach()
        # compute the descriptors for each of the sampled layouts
        # NB: this is split into minibatches since some rooms may be extremely large with many sampled layouts
        reference_descs = []
        for layout_minibatch in reference_layouts.split(
            self.config.TEST.LAYOUTS_MAX_BATCH
        ):
            layout_minibatch = layout_minibatch.unsqueeze(0).contiguous().cuda()
            reference_descs.append(self.forward(r=layout_minibatch)[0].detach())
        reference_descs = torch.cat(reference_descs)
        # compute the distances between each of the room panos and the grid of layouts
        n, _ = query_desc.shape
        reference_descs = reference_descs.unsqueeze(0).expand(n, -1, -1)
        pred_distances = (
            self.compute_distances(query_desc, reference_descs).detach().cpu()
        )

        # gather ranking info for the prediction vs the actual
        pose_distances = torch.norm(
            (query_pose.unsqueeze(1) - reference_poses.expand(n, -1, -1)), dim=-1
        )
        ranking_prediction = torch.argsort(pred_distances, dim=-1)[:, :5]
        ranking_layout = torch.argsort(gt_distances, dim=-1)[:, :5]
        ranking_pose = torch.argsort(pose_distances, dim=-1)[:, :5]

        retrieval_error = pose_distances.gather(
            1, ranking_prediction[:, 0].unsqueeze(1)
        )
        oracle_error = pose_distances.gather(1, ranking_layout[:, 0].unsqueeze(1))

        nn_poses = reference_poses[ranking_prediction[:, 0]]

        # vogel disc refinement
        if self.config.TEST.VOGEL_DISC_REFINE:
            refined_poses = self.vogel_refinement(
                query_desc, nn_poses, batch["geometry"], batch["floor"],
            )
        else:
            refined_poses = nn_poses

        # latent pose optimisation
        if self.config.TEST.LATENT_POSE_OPTIMISATION:
            optimised_poses = self.latent_pose_optimisation(
                query_desc, refined_poses, batch["geometry"], batch["floor"],
            )
        else:
            optimised_poses = refined_poses

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

    def inference_epoch_end(self, outputs, log_key):
        predictions = []
        layouts = []
        poses = []
        retrieval_errors = []
        oracle_errors = []
        refined_errors = []
        optimised_errors = []
        pred_poses = []

        scene_idxs = []
        room_idxs = []
        for i, out in enumerate(outputs):
            predictions.extend(out["pred_rank"].unsqueeze(0))
            layouts.extend(out["layout_rank"].unsqueeze(0))
            poses.extend(out["pose_rank"].unsqueeze(0))
            retrieval_errors.extend(out["retrieval_error"].unsqueeze(0))
            oracle_errors.extend(out["oracle_error"].unsqueeze(0))
            refined_errors.extend(out["refined_error"].unsqueeze(0))
            optimised_errors.extend(out["optimised_error"].unsqueeze(0))
            pred_poses.extend(out["pred_pose"].unsqueeze(0))

            num_rooms = len(out["retrieval_error"])
            scene_idxs.extend([i] * num_rooms)
            room_idxs.extend(list(range(num_rooms)))

        predictions = torch.cat(predictions)
        layouts = torch.cat(layouts)
        poses = torch.cat(poses)
        retrieval_errors = torch.cat(retrieval_errors)
        oracle_errors = torch.cat(oracle_errors)
        refined_errors = torch.cat(refined_errors)
        optimised_errors = torch.cat(optimised_errors)
        pred_poses = torch.cat(pred_poses)

        scene_idxs = torch.tensor(scene_idxs)
        room_idxs = torch.tensor(room_idxs)

        layout_r_at_1 = (predictions[:, 0] == layouts[:, 0]).float().mean().item()
        pose_r_at_1 = (predictions[:, 0] == poses[:, 0]).float().mean().item()
        oracle_r_at_1 = (layouts[:, 0] == poses[:, 0]).float().mean().item()

        layout_r_at_5 = recall_at_n(5, predictions, layouts).item()
        pose_r_at_5 = recall_at_n(5, predictions, poses).item()
        oracle_r_at_5 = recall_at_n(5, layouts, poses).item()

        median_retrieval_error = torch.median(retrieval_errors).item()
        median_refined_error = torch.median(refined_errors).item()
        median_optimised_error = torch.median(optimised_errors).item()
        median_oracle_error = torch.median(oracle_errors).item()

        threshold_1cm = (optimised_errors < 10).float().mean().item()
        threshold_5cm = (optimised_errors < 50).float().mean().item()
        threshold_10cm = (optimised_errors < 100).float().mean().item()
        threshold_100cm = (optimised_errors < 1000).float().mean().item()

        if self.config.TEST.METRIC_DUMP:
            data = {
                "scene_idxs": scene_idxs,
                "room_idxs": room_idxs,
                "oracle": oracle_errors.cpu(),
                "refinement": refined_errors.cpu(),
                "optimisation": optimised_errors.cpu(),
                "retrieval": retrieval_errors.cpu(),
                "pred_poses": pred_poses,
            }
            torch.save(data, self.config.TEST.METRIC_DUMP)

        stats_to_log = {
            "{}/layout_r_at_1".format(log_key): layout_r_at_1,
            "{}/pose_r_at_1".format(log_key): pose_r_at_1,
            "{}/layout_r_at_5".format(log_key): layout_r_at_5,
            "{}/pose_r_at_5".format(log_key): pose_r_at_5,
            "{}/oracle_r_at_1".format(log_key): oracle_r_at_1,
            "{}/oracle_r_at_5".format(log_key): oracle_r_at_5,
            "{}/median_retrieval_error".format(log_key): median_retrieval_error,
            "{}/median_refined_error".format(log_key): median_refined_error,
            "{}/median_optimised_error".format(log_key): median_optimised_error,
            "{}/median_oracle_error".format(log_key): median_oracle_error,
            "{}/threshold_1cm".format(log_key): threshold_1cm,
            "{}/threshold_5cm".format(log_key): threshold_5cm,
            "{}/threshold_10cm".format(log_key): threshold_10cm,
            "{}/threshold_100cm".format(log_key): threshold_100cm,
        }
        return {"test_loss": 1 - layout_r_at_1, "log": stats_to_log}

    def train_dataloader(self):
        return build_dataloader(self.config, "train")

    def val_dataloader(self):
        return build_dataloader(self.config, "val")

    def test_dataloader(self):
        # Convenient to make "test" actually the validation set so you can recheck val acc at any point
        if self.config.TEST.VAL_AS_TEST:
            return build_dataloader(self.config, "val")
        return build_dataloader(self.config, "test")

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.inference_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        return self.inference_epoch_end(outputs, "test")


class Layout2LayoutBase(Image2LayoutBase):
    def __init__(self, config):
        super(Layout2LayoutBase, self).__init__(config)
        self.embedder = LayoutModule(config)
        self.reference_embedder = None
        self.query_embedder = None
        self.desc_length = config.MODEL.DESC_LENGTH
        self.config = config
        self.query_key = "pano_layout"

    def forward(self, q=None, r=None):
        if q is None and r is None:
            raise Exception

        if q is not None:
            q = self.embedder(q)
            if r is None:
                return q

        if r is not None:
            n, m, c, h, w = r.shape
            r = r.reshape(n * m, c, h, w)
            r = self.embedder(r)
            r = r.reshape(n, m, self.desc_length)
            if q is None:
                return r

        d = self.compute_distances(q, r)
        return d
