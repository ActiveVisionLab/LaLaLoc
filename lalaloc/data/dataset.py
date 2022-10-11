import os
import random
import warnings
from math import dist

import numpy as np
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
from pytorch3d.structures import Pointclouds
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils.chamfer import chamfer
from ..utils.floorplan import pose_to_pixel_loc, sample_locs
from ..utils.projection import (
    project_depth_to_pc,
    project_depth_to_pc_batched,
    projects_onto_floor,
)
from ..utils.render import (
    render_scene,
    render_scene_batched,
    render_semantic_batched,
    render_semantics,
)
from .load import (
    create_floorplan_from_annos,
    load_scene_annos,
    prepare_geometry_from_annos,
)
from .split import scenes_split
from .transform import build_transform


def sample_xy_displacement(max_dist=1, min_dist=0):
    radius = random.uniform(min_dist, max_dist)
    angle = random.uniform(0, 2 * np.pi)
    x = radius * np.sin(angle)
    y = radius * np.cos(angle)
    return np.array([x, y, 0]) * 1000


def load_scene_data(
    scene_ids, root_path, for_visualisation=False, pix_per_mm=0.025, min_factor=32
):
    scenes = []
    for scene_id in tqdm(scene_ids):
        annos = load_scene_annos(root_path, scene_id)
        floorplan, plan_params = create_floorplan_from_annos(
            annos, scene_id, pix_per_mm, min_factor
        )
        scene_geometry, floor_planes, limits = prepare_geometry_from_annos(
            annos, for_visualisation=for_visualisation
        )
        scene_path = os.path.join(root_path, f"scene_{scene_id:05d}", "2D_rendering")

        scene_rooms = []
        for room_id in np.sort(os.listdir(scene_path)):
            room_path = os.path.join(scene_path, room_id, "panorama")
            panorama_path = os.path.join(room_path, "{}", "rgb_{}light.png")
            pose = np.loadtxt(os.path.join(room_path, "camera_xyz.txt"))

            scene_rooms.append(
                {
                    "id": room_id,
                    "path": room_path,
                    "panorama": panorama_path,
                    "pose": pose,
                }
            )
        scenes.append(
            {
                "id": scene_id,
                "geometry": scene_geometry,
                "rooms": scene_rooms,
                "floor_planes": floor_planes,
                "limits": limits,
                "floorplan": floorplan,
                "floorplan_params": plan_params,
            }
        )
    return scenes


class Structured3DPlans(Dataset):
    def __init__(
        self, config, split="train", visualise=False,
    ):
        scene_ids = scenes_split(split)
        dataset_path = config.DATASET.PATH
        self.scenes = load_scene_data(
            scene_ids,
            dataset_path,
            visualise,
            config.DATASET.PIX_PER_MM,
            config.DATASET.FLOORPLAN_DIVISIBLE_BY,
        )
        self.is_train = split == "train"
        self.visualise = visualise
        self.transform = build_transform(config, self.is_train)
        self.layout_transform = build_transform(config, self.is_train, is_layout=True)
        self.augment = config.DATASET.AUGMENT_LAYOUTS

        self.precomputed = None

        if self.is_train:
            furniture_levels = config.DATASET.TRAIN_FURNITURE
            lighting_levels = config.DATASET.TRAIN_LIGHTING
        else:
            furniture_levels = config.DATASET.TEST_FURNITURE
            lighting_levels = config.DATASET.TEST_LIGHTING
        self.furniture_levels = furniture_levels
        self.lighting_levels = lighting_levels
        self.compute_gt_dist = config.TRAIN.COMPUTE_GT_DIST
        self.compute_gt_dist_test = config.TEST.COMPUTE_GT_DIST
        self.config = config

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        data = self.scenes[idx]
        geometry = data["geometry"]
        rooms = data["rooms"]
        floor = data["floor_planes"]
        limits = data["limits"]
        floorplan = data["floorplan"]
        floorplan_params = data["floorplan_params"]

        if self.is_train:
            # randomly select a room from the scene
            room = random.choice(rooms)
            # sample and process the +ve and -ve traning examples
            sampled_poses_and_rooms = self._sample_train_poses(
                room["pose"], floor, limits, scene_id=data["id"]
            )
            sampled_layouts = render_scene_batched(
                self.config, geometry, sampled_poses_and_rooms
            )
            sampled_pointclouds = project_depth_to_pc_batched(
                self.config, sampled_layouts
            )
            room_data = self._process_room(
                data["id"], room, geometry, floor, limits, sampled_pointclouds,
            )
            panorama = room_data["image"]
            pano_pose = room_data["pose"]
            pano_depths = room_data["layout"]
            pano_layouts = self.layout_transform(room_data["layout"])
            pano_room_idx = [room_data["room_idx"]]
            # in some scenarios we do not need the gt distances
            if self.compute_gt_dist:
                distances = room_data["gt_distances"]
            else:
                distances = []
            # semantics are only used for visualation therefore not needed here
            sampled_semantics = [np.empty((0, 0))]
            pano_semantics = [np.empty((0, 0))]
        else:
            sampled_poses_and_rooms = self._sample_test_poses(
                limits,
                rooms[0]["pose"][-1],
                floor,
                step=self.config.TEST.POSE_SAMPLE_STEP,
            )
            # if the grid size is set to be very large sometimes there are no test poses
            if sampled_poses_and_rooms:
                sampled_layouts = render_scene_batched(
                    self.config, geometry, sampled_poses_and_rooms
                )
                sampled_semantics = render_semantic_batched(
                    self.config, geometry, sampled_poses_and_rooms
                )
                sampled_pointclouds = project_depth_to_pc_batched(
                    self.config, sampled_layouts
                )
            else:
                warnings.warn(
                    "The grid size is set too large for the scene leading to there being no valid test poses."
                )
                sampled_layouts = None
                sampled_semantics = None
                sampled_pointclouds = []

            panoramas = []
            pano_poses = []
            pano_layouts = []
            pano_room_idx = []
            pano_semantics = []
            distances_all = []
            for room in rooms:
                room_data = self._process_room(
                    data["id"], room, geometry, floor, limits, sampled_pointclouds,
                )
                panoramas.append(room_data["image"])
                pano_poses.append(room_data["pose"])
                pano_layouts.append(room_data["layout"])
                pano_room_idx.append(room_data["room_idx"])
                pano_semantics.append(room_data["semantics"])
                distances_all.append(room_data["gt_distances"])
            panorama = torch.stack(panoramas)
            pano_pose = np.stack(pano_poses)
            pano_room_idx = torch.Tensor(pano_room_idx)
            pano_depths = np.stack(pano_layouts)
            pano_layouts = torch.stack(
                [self.layout_transform(l) for l in pano_layouts], dim=0
            )
            if self.compute_gt_dist_test:
                distances = torch.stack(distances_all)
            else:
                distances = []
        pano_semantics = np.stack(pano_semantics)

        sampled_depths = np.stack(sampled_layouts)
        sampled_semanitcs = np.stack(sampled_semantics)
        sampled_layouts = torch.stack(
            [self.layout_transform(l) for l in sampled_layouts], dim=0
        )

        pano_pose = torch.Tensor(pano_pose)
        sampled_poses = torch.Tensor([p for p, _ in sampled_poses_and_rooms])
        sampled_room_idxs = torch.Tensor([r for _, r in sampled_poses_and_rooms])

        if self.is_train:
            # geometry doesn't support batching but isn't used in training
            geometry = []
            floor = []

        floorplan = tvf.to_tensor(floorplan)

        return {
            "panorama": panorama,
            "pano_layout": pano_layouts,
            "pano_pose": pano_pose,
            "pano_room_idx": pano_room_idx,
            "pano_depths": pano_depths,
            "pano_semantics": pano_semantics,
            "sampled_layouts": sampled_layouts,
            "sampled_poses": sampled_poses,
            "sampled_room_idxs": sampled_room_idxs,
            "sampled_depths": sampled_depths,
            "sampled_semantics": sampled_semanitcs,
            "distances": distances,
            "geometry": geometry,
            "floor": floor,
            "floorplan": floorplan,
            "floorplan_params": floorplan_params,
        }

    def _process_room(
        self, scene_id, room, geometry, floor, limits, pointclouds_layouts
    ):
        pose_pano, layout_pano, semantics_pano, room_idx = self._get_room_data(
            room, geometry, floor, scene_id
        )
        pointcloud_pano = torch.Tensor(project_depth_to_pc(self.config, layout_pano))
        pointclouds_pano = Pointclouds([pointcloud_pano,] * len(pointclouds_layouts))

        if (self.is_train and self.compute_gt_dist) or (
            not self.is_train and self.compute_gt_dist_test
        ):
            distances = chamfer(pointclouds_pano, pointclouds_layouts)
        else:
            distances = []

        furniture = random.choice(self.furniture_levels)
        lighting = random.choice(self.lighting_levels)
        panorama_path = room["panorama"].format(furniture, lighting)
        # sometimes the panorama image can be get corrupted
        # if this happens, reextract the relevant zip file
        try:
            panorama = Image.open(panorama_path).convert("RGB")
        except Exception as e:
            print(panorama_path)
            print(e)

        panorama = self.transform(panorama)

        return {
            "image": panorama,
            "pose": pose_pano,
            "layout": layout_pano,
            "semantics": semantics_pano,
            "gt_distances": distances,
            "room_idx": room_idx,
        }

    def _sample_train_poses(
        self, pose_panorama, floor_geometry, limits, offset=500, scene_id=None
    ):
        poses = []
        for _ in range(self.config.TRAIN.NUM_NEAR_SAMPLES):
            near_room_idx = -1
            num_attempts = 0
            while near_room_idx < 0:
                if num_attempts > 100:
                    pose_near = pose_panorama
                else:
                    pose_near = pose_panorama + sample_xy_displacement(
                        max_dist=self.config.TRAIN.NEAR_MAX_DIST,
                        min_dist=self.config.TRAIN.NEAR_MIN_DIST,
                    )
                near_room_idx = projects_onto_floor(pose_near, floor_geometry)
                num_attempts += 1
            poses.append((pose_near, near_room_idx))

        for _ in range(self.config.TRAIN.NUM_FAR_SAMPLES):
            far_room_idx = -1
            num_attempts = 0
            while far_room_idx < 0:
                x_location = np.random.uniform(
                    limits[0] + (offset / 2), limits[1] - (offset / 2)
                )
                y_location = np.random.uniform(
                    limits[2] + (offset / 2), limits[3] - (offset / 2)
                )
                pose_far = np.array([x_location, y_location, pose_panorama[-1]])

                if (
                    np.linalg.norm(pose_panorama - pose_far)
                    < self.config.TRAIN.FAR_MIN_DIST
                ):
                    continue
                far_room_idx = projects_onto_floor(pose_far, floor_geometry)
                if num_attempts >= 100:
                    pose_far = pose_near
                far_room_idx = projects_onto_floor(pose_far, floor_geometry)
                num_attempts += 1
            poses.append((pose_far, far_room_idx))

        if self.config.TRAIN.APPEND_GT:
            gt_room_idx = projects_onto_floor(pose_panorama, floor_geometry)
            poses.append((pose_panorama, gt_room_idx))
        return poses

    def _sample_test_poses(self, limits, z, floor_geometry, step=1000):
        x_locations = np.arange(limits[0] + (step / 2), limits[1] + (step / 2), step)
        y_locations = np.arange(limits[2] + (step / 2), limits[3] + (step / 2), step)

        poses = np.meshgrid(x_locations, y_locations)
        poses = np.stack(poses, axis=2).reshape(-1, 2)
        poses = np.concatenate([poses, np.full((poses.shape[0], 1), z)], axis=1)

        room_idxs = [projects_onto_floor(pose, floor_geometry) for pose in poses]
        pose_grid = [(p, i) for p, i in zip(poses, room_idxs) if i >= 0]
        return pose_grid

    def _get_room_data(self, room, geometry, floor, scene_id):
        pose_pano = room["pose"]
        room_idx = projects_onto_floor(pose_pano, floor)
        if room_idx < 0:
            warnings.warn("pose outside room: {}".format(scene_id))
        layout_pano = render_scene(self.config, geometry[room_idx], pose_pano)
        semantics_pano = render_semantics(self.config, geometry[room_idx], pose_pano)
        return pose_pano, layout_pano, semantics_pano, room_idx


class TargetEmbeddingDataset(Structured3DPlans):
    def __init__(self, encoder, config, split="train", visualise=False, device="cpu:0"):
        super().__init__(config, split=split, visualise=visualise)

        with torch.no_grad():
            encoder.eval()
            for scene in tqdm(self.scenes):
                floorplan = scene["floorplan"]
                floorplan_params = scene["floorplan_params"]
                scale = torch.tensor([floorplan_params["scale"]])
                shift = torch.tensor(floorplan_params["shift"])
                plan_height = floorplan_params["h"]
                plan_width = floorplan_params["w"]
                floorplan = floorplan[:plan_height, :plan_width]

                floorplan = tvf.to_tensor(floorplan).to(device).unsqueeze(0)
                plan_embed = encoder(floorplan).cpu()

                subsample_x = self.config.TEST.SUBSAMPLE_PLAN_X
                if subsample_x > 1:
                    plan_embed = plan_embed[:, :, ::subsample_x, ::subsample_x]
                    floorplan = floorplan[:, :, ::subsample_x, ::subsample_x]
                    scale = scale / subsample_x

                query_poses = []
                rooms = scene["rooms"]
                for room in rooms:
                    query_poses.append(room["pose"])
                query_poses = torch.Tensor(np.stack(query_poses))
                query_locs = pose_to_pixel_loc(query_poses.unsqueeze(0), scale, shift)

                target_embeddings = sample_locs(
                    plan_embed, query_locs, normalise=self.config.MODEL.NORMALISE_SAMPLE
                ).squeeze(0)

                scene["embeddings"] = target_embeddings

    def __getitem__(self, idx):
        data = self.scenes[idx]
        rooms = data["rooms"]
        embeddings = data["embeddings"]

        room_idx = random.randrange(0, len(rooms))
        room = rooms[room_idx]

        furniture = random.choice(self.furniture_levels)
        lighting = random.choice(self.lighting_levels)
        panorama_path = room["panorama"].format(furniture, lighting)
        # sometimes the panorama image can be get corrupted
        # if this happens, reextract the relevant zip file
        try:
            panorama = Image.open(panorama_path).convert("RGB")
        except Exception as e:
            print(panorama_path)
            print(e)

        panorama = self.transform(panorama)
        embedding = embeddings[room_idx]
        return {
            "panorama": panorama,
            "target_embedding": embedding,
        }
