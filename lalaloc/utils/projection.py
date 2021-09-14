import numpy as np
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.mesh.rasterize_meshes import barycentric_coordinates

from .panorama import coords_to_uv, uvs_to_rays


def project_depth_to_pc(
    config, depth, camera_type="panorama", camera_params=None,
):
    height, width = depth.shape
    depth = depth.reshape(-1)
    xys_camera = np.stack(
        np.meshgrid(np.arange(width), np.arange(height)), axis=2
    ).reshape((-1, 2))

    if camera_type == "panorama":
        uvs = coords_to_uv(xys_camera, width, height)
        rays = uvs_to_rays(uvs).astype(np.float32).reshape(-1, 3)
    else:
        raise NotImplementedError(
            "{} camera_type is not currently implemented".format(camera_type)
        )
    invalid_depth = np.isnan(depth) | (depth <= 0)
    points = depth[~invalid_depth].reshape(-1, 1) * rays[~invalid_depth]

    return points


def project_depth_to_pc_batched(
    config, depths, camera_type="panorama", camera_params=None
):
    height, width = config.RENDER.IMG_SIZE
    xys_camera = np.stack(
        np.meshgrid(np.arange(width), np.arange(height)), axis=2
    ).reshape((-1, 2))

    if camera_type == "panorama":
        uvs = coords_to_uv(xys_camera, width, height)
        rays = uvs_to_rays(uvs).astype(np.float32).reshape(-1, 3)
    else:
        raise NotImplementedError(
            "{} camera_type is not currently implemented".format(camera_type)
        )
    points = []
    for depth in depths:
        depth = depth.reshape(-1)
        invalid_depth = np.isnan(depth) | (depth <= 0)
        point = depth[~invalid_depth].reshape(-1, 1) * rays[~invalid_depth]
        points.append(torch.Tensor(point))

    points = Pointclouds(points)
    return points


def projects_onto_floor(location, floors):
    for idx, (verts, faces) in enumerate(zip(floors.verts_list(), floors.faces_list())):
        triangles = [verts[idxs] for idxs in faces]
        for triangle in triangles:
            bary = barycentric_coordinates(location, *triangle)
            bary = torch.Tensor(bary)
            if (bary >= 0).all() and (bary <= 1).all() and abs(sum(bary) - 1) < 1e-5:
                return idx
    return -1
