"""
Parts of this code are modified from: https://github.com/bertjiazheng/Structured3D
Copyright (c) 2019 Structured3D Group
"""
import numpy as np


def uvs_to_rays(uvs):
    xs_ray = np.cos(uvs[:, 1]) * np.sin(uvs[:, 0])
    ys_ray = np.cos(uvs[:, 1]) * np.cos(uvs[:, 0])
    zs_ray = np.sin(uvs[:, 1])

    rays = np.stack([xs_ray, ys_ray, zs_ray], axis=1)
    rays = rays / np.linalg.norm(rays, axis=1).reshape(-1, 1)
    return rays


def coords_to_uv(coords, width, height):
    """
    Image coordinates (xy) to uv
    """
    middleX = width / 2 + 0.5
    middleY = height / 2 + 0.5
    uv = np.hstack(
        [
            (coords[:, [0]] - middleX) / width * 2 * np.pi,
            -(coords[:, [1]] - middleY) / height * np.pi,
        ]
    )
    return uv
