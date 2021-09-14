import torch

def triplet_loss(distances, margin=1):
    pos_distances = distances[:, 0]
    neg_distances = distances[:, 1]

    losses = (pos_distances - neg_distances + margin).clamp(min=0)
    loss = losses.sum()
    if loss > 0:
        loss = loss / len(torch.nonzero(losses))
    return loss


def bbs_loss(distances, distances_truth):
    n = distances.shape[1]
    rows = distances.unsqueeze(1).expand(-1, n, -1)
    cols = distances.unsqueeze(2).expand(-1, -1, n)

    rows_gt = distances_truth.unsqueeze(1).expand(-1, n, -1)
    cols_gt = distances_truth.unsqueeze(2).expand(-1, -1, n)

    loss = (rows / cols).log() - (rows_gt / cols_gt).log()
    # remove i, i matches
    identity = torch.eye(n, device=loss.device).unsqueeze(0).expand_as(loss).bool()
    loss = loss[~identity]

    loss = (loss ** 2).mean() / 2
    return loss