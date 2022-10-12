import torch
import torch.nn.functional as F


def pose_to_pixel_loc(pose, scale, shift):
    locs = pose.clone()[:, :, :2]
    locs -= shift.view(-1, 1, 2)
    locs *= scale.view(-1, 1, 1)
    return locs


def pixel_loc_to_pose(locs, scale, shift, z):
    poses = locs.clone().float()
    poses /= scale.view(-1, 1, 1)
    poses += shift.view(-1, 1, 2)
    h, w, _ = poses.shape
    zs = torch.full((h, w, 1), z).to(poses.device)
    return torch.cat([poses, zs], dim=-1)


def create_pixel_loc_grid(w, h):
    x = torch.arange(w)
    y = torch.arange(h)
    ys, xs = torch.meshgrid([y, x])
    loc_grid = torch.stack([xs, ys], dim=-1)
    return loc_grid


def sample_locs(tensor, locs, normalise=True):
    b, c, h, w = tensor.shape
    locs[:, :, 0] = locs[:, :, 0] / (w / 2) - 1
    locs[:, :, 1] = locs[:, :, 1] / (h / 2) - 1
    _, n, _ = locs.shape
    locs = locs.view(b, 1, -1, 2).clone().float()
    sampled = F.grid_sample(tensor, locs, align_corners=False)
    sampled = sampled.view(b, c, n).permute(0, 2, 1)
    if normalise:
        sampled = F.normalize(sampled, dim=-1)
    return sampled
