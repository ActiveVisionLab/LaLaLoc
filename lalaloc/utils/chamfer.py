import torch
from pytorch3d.loss import chamfer_distance


def chamfer(points1, points2, device=None):
    with torch.no_grad():
        if device is None:
            points1 = points1.cuda()
            points2 = points2.cuda()
        else:
            points1 = points1.to(device)
            points2 = points2.to(device)
        distances, _ = chamfer_distance(points1, points2, batch_reduction=None)
    return distances.cpu()
