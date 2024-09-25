import torch

def lddt(pred_points, true_points, points_mask, cutoff=15.):
    """Computes the lddt score for a batch of coordinates.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896
        Inputs: 
        * pred_coords: (b, l, d) array of predicted 3D points.
        * true_points: (b, l, d) array of true 3D points.
        * points_mask : (b, l) binary-valued array. 1 for points that exist in
            the true points
        * cutoff: maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt scores ranging between 0 and 1
    """
    assert len(pred_points.shape) == 3 and pred_points.shape[-1] == 3
    assert len(true_points.shape) == 3 and true_points.shape[-1] == 3

    eps = 1e-10

    # Compute true and predicted distance matrices. 
    pred_cdist = torch.cdist(pred_points, pred_points, p=2)
    true_cdist = torch.cdist(true_points, true_points, p=2)

    cdist_to_score = ((true_cdist < cutoff) *
            (rearrange(points_mask, 'b i -> b i ()')*rearrange(points_mask, 'b j -> b () j')) *
            (1.0 - torch.eye(true_cdist.shape[1], device=points_mask.device)))  # Exclude self-interaction

    # Shift unscored distances to be far away
    dist_l1 = torch.abs(true_cdist - pred_cdist)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * sum([dist_l1 < t for t in (0.5, 1.0, 2.0, 4.0)])

    # Normalize over the appropriate axes.
    return (torch.sum(cdist_to_score * score, dim=-1) + eps)/(torch.sum(cdist_to_score, dim=-1) + eps)

def plddt(logits):
    """Compute per-residue pLDDT from logits
    """
    device = logits.device if hasattr(logits, 'device') else None
    # Shape (b, l, c)
    b, c = logits.shape[0], logits.shape[-1]
    width = 1.0 / c
    centers = torch.arange(start=0.5*width, end=1.0, step=width, device=device)
    probs = F.softmax(logits, dim=-1)
    return torch.einsum('c,... c -> ...', centers, probs)

