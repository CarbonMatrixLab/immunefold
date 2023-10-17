import torch
from torch.nn import functional as F

def compute_plddt(logits):
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = torch.arange(start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device)
  probs = F.softmax(logits, dim=-1)
  predicted_lddt_ca = torch.sum(probs * bin_centers[None, None, :], axis=-1)

  return predicted_lddt_ca * 100
