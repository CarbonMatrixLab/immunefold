import functools

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from carbonmatrix.model import r3
from carbonmatrix.common import residue_constants
from carbonmatrix.trainer.base_loss import registry_loss
from carbonmatrix.model.utils import (
        l2_normalize,
        squared_difference,
        batched_select,
        lddt)

@registry_loss
def seq_mask_loss(batch, values, config):
    labels, label_mask = batch['label_esm_seq'][:,1:-1], batch['label_esm_mask'][:,1:-1]
    logits = values['esm_logits']
    eps = 1e-8

    logits = rearrange(logits, 'b l c -> b c l')

    loss = F.cross_entropy(logits, labels, reduction='none')

    loss = torch.sum(loss * label_mask) / (torch.sum(label_mask) + eps)

    return dict(loss=loss)

@registry_loss
def distogram_loss(batch, value, config):
    """Log loss of a distogram."""
    c = config
    value = value['heads']['distogram']

    logits, breaks = value['logits'], value['breaks']
    assert len(logits.shape) == 4
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']

    assert positions.shape[-1] == 3

    sq_breaks = torch.square(breaks)

    dist2 = torch.sum(
        torch.square(
            rearrange(positions, 'b l c -> b l () c') -
            rearrange(positions, 'b l c -> b () l c')),
        dim=-1,
        keepdims=True)

    true_bins = torch.sum(dist2 > sq_breaks, dim=-1)

    errors = -torch.sum(F.one_hot(true_bins, logits.shape[-1]) * F.log_softmax(logits, dim=-1), dim=-1)

    square_mask = rearrange(mask, 'b l -> b () l') * rearrange(mask, 'b l -> b l ()')

    avg_error = (
        torch.sum(errors * square_mask) /
        (1e-6 + torch.sum(square_mask)))
    return dict(loss=avg_error, true_dist=torch.sqrt(dist2+1e-6))

@registry_loss
def folding_loss(batch, value, config):
    c = config
    assert 'structure_module' in value['heads']
    value = value['heads']['structure_module']

    backbone_fape_loss = compute_backbone_loss(batch, value, config)

    chi_loss, chi_loss_items= compute_chi_loss(batch, value, config)

    if 'renamed_atom14_gt_positions' not in value:
        value.update(compute_renamed_ground_truth(batch, value['final_atom14_positions']))

    sc_fape_loss = compute_sidechain_loss(batch, value, config)

    '''
    violation_loss, violation_loss_items = between_residue_bond_loss(
            value['final_atom14_positions'],
            batch['atom14_atom_exists'],
            batch['chain_id'],
            batch['seq'])
    '''

    loss = (
            c.sidechain_fape_weight * sc_fape_loss +
            c.backbone_fape_weight * backbone_fape_loss +
            c.chi_weight * chi_loss
            #c.structural_violation_loss_weight * violation_loss
            )

    return dict(loss=loss,
            backbone_fape_loss=backbone_fape_loss,
            sidechain_fape_loss=sc_fape_loss,
            chi_loss=chi_loss,
            #violation_loss=violation_loss,
            **chi_loss_items,
            #**violation_loss_items
            )

def predicted_lddt_loss(batch, value, config):
    c = config
    logits = value['heads']['predicted_lddt']['logits']

    pred_all_atom_pos = value['heads']['structure_module']['final_atom14_positions']
    true_all_atom_pos = batch['atom14_gt_positions']
    all_atom_mask = batch['atom14_gt_exists']

    num_bins = logits.shape[-1]

    with torch.no_grad():
        lddt_ca = lddt(
                pred_all_atom_pos[:,:,1],
                true_all_atom_pos[:,:,1],
                all_atom_mask[:,:,1]).detach()

    bin_index = torch.clip(torch.floor(lddt_ca * num_bins).long(), max=num_bins - 1)

    lddt_ca_one_hot = F.one_hot(bin_index, num_bins).to(dtype=logits.dtype)

    errors = -torch.sum(lddt_ca_one_hot * F.log_softmax(logits, dim=-1), dim=-1)

    mask_ca = all_atom_mask[:,:,1]

    loss = torch.sum(errors * mask_ca) / (1e-6 + torch.sum(mask_ca))

    return dict(loss=loss)

def compute_chi_loss(batch, value, config):
    device = batch['seq'].device
    eps = 1e-10

    num_chi_iter = float(len(value['sidechains']))

    pred_angles = torch.stack([v['angles_sin_cos'] for v in value['sidechains']], dim=-3)[...,3:,:]

    sin_cos_true_chi = batch['torsion_angles_sin_cos'][...,3:,:][...,None,:,:]

    chi_pi_periodic = batched_select(torch.tensor(residue_constants.chi_pi_periodic, device=device), batch['seq'])
    shifted_mask = (1 - 2 * chi_pi_periodic)[...,None,:,None]
    sin_cos_true_chi_shifted = shifted_mask * sin_cos_true_chi

    sq_chi_error = torch.sum(squared_difference(sin_cos_true_chi, pred_angles), dim=-1)
    sq_chi_error_shifted = torch.sum(squared_difference(sin_cos_true_chi_shifted, pred_angles), dim=-1)

    #sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    sq_chi_error = torch.min(sq_chi_error, sq_chi_error_shifted)

    chi_mask = batch['torsion_angles_mask'][..., None, 3:]
    sq_chi_loss = torch.sum(chi_mask * sq_chi_error) / (torch.sum(chi_mask) * num_chi_iter + eps)

    unnormed_angles = torch.stack([v['unnormalized_angles_sin_cos'] for v in value['sidechains']], dim=-3)[...,3:,:]

    angle_norm = torch.sqrt(torch.sum(torch.square(unnormed_angles), dim=-1) + eps)
    norm_error = torch.abs(angle_norm - 1.)
    angle_norm_mask = batch['mask'][...,None] * batched_select(torch.tensor(residue_constants.chi_angles_mask, device=device), batch['seq'])
    angle_norm_mask = angle_norm_mask[...,None,:]
    angle_norm_loss = torch.sum(angle_norm_mask * norm_error) / (torch.sum(angle_norm_mask) * num_chi_iter  + eps)

    loss = sq_chi_loss + config.angle_norm_weight * angle_norm_loss

    return loss, dict(
            sq_chi_loss=sq_chi_loss,
            angle_norm_loss=angle_norm_loss
            )

def compute_sidechain_loss(batch, value, config):
    alt_naming_is_better = value['alt_naming_is_better']
    renamed_gt_frames_rots = (
            (1. - alt_naming_is_better[:, :, None, None, None])
            * batch['rigidgroups_gt_frames'][0]
            + alt_naming_is_better[:, :, None, None, None]
            * batch['rigidgroups_alt_gt_frames'][0])

    renamed_gt_frames_trans = (
            (1. - alt_naming_is_better[:, :, None, None])
            * batch['rigidgroups_gt_frames'][1]
            + alt_naming_is_better[:, :, None, None]
            * batch['rigidgroups_alt_gt_frames'][1])

    renamed_gt_frames = (renamed_gt_frames_rots, renamed_gt_frames_trans)

    pred_frames = value['sidechains'][-1]['frames']
    pred_positions = value['sidechains'][-1]['atom_pos']

    fape = frame_aligned_point_error(
        pred_frames=pred_frames,
        target_frames=renamed_gt_frames,
        frames_mask=batch['rigidgroups_gt_exists'],
        pred_positions=pred_positions,
        target_positions=value['renamed_atom14_gt_positions'],
        positions_mask=value['renamed_atom14_gt_exists'],
        clamp_distance=config.fape.clamp_distance,
        length_scale=config.fape.loss_unit_distance,
        unclamped_ratio=config.fape.unclamped_ratio)

    return fape

def compute_final_backbone_loss(batch, value, config):

    atom_n, frame_n = 5, 4

    pred_frames = r3.rigids_op(value['sidechains'][-1]['frames'], lambda x: x[:,:,:frame_n])
    pred_positions = value['sidechains'][-1]['atom_pos'][:,:,:atom_n]

    gt_frames = r3.rigids_op(batch['rigidgroups_gt_frames'], lambda x: x[:,:,:frame_n])
    gt_frame_mask = batch['rigidgroups_group_exists'][:,:,:frame_n]
    gt_positions = batch['atom14_gt_positions'][:,:,:atom_n]
    gt_position_mask = batch['atom14_gt_exists'][:,:,:atom_n]

    fape = frame_aligned_point_error(
        pred_frames=pred_frames,
        target_frames=gt_frames,
        frames_mask=gt_frame_mask,
        pred_positions=pred_positions,
        target_positions=gt_positions,
        positions_mask=gt_position_mask,
        clamp_distance=config.fape.clamp_distance,
        length_scale=config.fape.loss_unit_distance,
        unclamped_ratio=config.fape.unclamped_ratio)

    return fape

def compute_backbone_loss(batch, value, config):
    assert 'traj' in value
    c = config

    target_positions, positions_mask = batch['atom14_gt_positions'], batch['atom14_gt_exists']
    target_frames, frames_mask = batch['rigidgroups_gt_frames'], batch['rigidgroups_gt_exists']

    def _yield_backbone_loss(traj, pair_mask, clamp_distance, loss_unit_distance, pos_weight):
        target_backbone_frames = tuple(map(lambda x : x[:, :, 0], target_frames))
        for pred_frames in traj:
            rots, trans = pred_frames
            r = frame_aligned_point_error(
                    pred_frames=pred_frames,
                    target_frames=target_backbone_frames,
                    frames_mask=frames_mask[:,:,0],
                    pred_positions=trans,
                    target_positions=target_positions[:,:,1],
                    positions_mask=positions_mask[:,:,1],
                    pos_weight = pos_weight,
                    clamp_distance=clamp_distance,
                    length_scale=loss_unit_distance,
                    unclamped_ratio=c.fape.unclamped_ratio)
            yield r

    traj = value['traj']

    if c.fape.loop_weight.enabled:
        loop_mask = torch.eq(batch['cdr_def'], 5)
        loop_mask = torch.logical_or(loop_mask[:,:,None], loop_mask[:,None,:]).to(
                dtype=target_positions.dtype, device=target_positions.device)
        loop_weight = (1. - loop_mask) + loop_mask * c.local_fape.loop_weight.weight
    else:
        loop_weight = None

    fape = sum(_yield_backbone_loss(
        traj, pair_mask=None,
        clamp_distance=c.fape.clamp_distance,
        loss_unit_distance=c.fape.loss_unit_distance,
        pos_weight=loop_weight)) / len(traj)

    return fape

def find_optimal_renaming(
    atom14_gt_positions, atom14_alt_gt_positions,
    atom14_atom_is_ambiguous,
    atom14_gt_exists,
    atom14_pred_positions,
    atom14_atom_exists):

    assert atom14_gt_positions.ndim == 4
    assert atom14_alt_gt_positions.ndim == 4
    assert atom14_atom_is_ambiguous.ndim == 3
    assert atom14_gt_exists.ndim == 3
    assert atom14_pred_positions.ndim == 4
    assert atom14_atom_exists.ndim == 3

    # shape (N, N, 14, 14)
    pred_dists = torch.sqrt(1e-10 + torch.sum(
        squared_difference(
            atom14_pred_positions[:, :, None, :, None, :],
            atom14_pred_positions[:, None, :, None, :, :]),
        axis=-1))

    # shape (N, N, 14, 14)
    gt_dists = torch.sqrt(1e-10 + torch.sum(
        squared_difference(
            atom14_gt_positions[:, :, None, :, None, :],
            atom14_gt_positions[:, None, :, None, :, :]),
        axis=-1))
    alt_gt_dists = torch.sqrt(1e-10 + torch.sum(
        squared_difference(
            atom14_alt_gt_positions[:, :, None, :, None, :],
            atom14_alt_gt_positions[:, None, :, None, :, :]),
        axis=-1))

    # shape (N, N, 14, 14)
    lddt = torch.sqrt(1e-10 + squared_difference(pred_dists, gt_dists))
    alt_lddt = torch.sqrt(1e-10 + squared_difference(pred_dists, alt_gt_dists))

    # shape (N ,N, 14, 14)
    mask = torch.logical_and(
            torch.logical_and(atom14_gt_exists[:, :, None, :, None], atom14_atom_is_ambiguous[:, :, None, :, None]),
            torch.logical_and(atom14_gt_exists[:, None, :, None, :], torch.logical_not(atom14_atom_is_ambiguous[:, None, :, None, :])))

    # shape (N)
    per_res_lddt = torch.sum(mask * lddt, dim=[2, 3, 4])
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=[2, 3, 4])

    # shape (N)
    alt_naming_is_better = torch.lt(alt_per_res_lddt, per_res_lddt).to(dtype=torch.float32)

    return alt_naming_is_better  # shape (N)

def compute_renamed_ground_truth(batch, atom14_pred_positions):
  alt_naming_is_better = find_optimal_renaming(
          atom14_gt_positions=batch['atom14_gt_positions'],
          atom14_alt_gt_positions=batch['atom14_alt_gt_positions'],
          atom14_atom_is_ambiguous=batch['atom14_atom_is_ambiguous'],
          atom14_gt_exists=batch['atom14_gt_exists'],
          atom14_pred_positions=atom14_pred_positions,
          atom14_atom_exists=batch['atom14_atom_exists'])

  renamed_atom14_gt_positions = (
          (1. - alt_naming_is_better[:, :, None, None])
          * batch['atom14_gt_positions']
          + alt_naming_is_better[:, :, None, None]
          * batch['atom14_alt_gt_positions'])

  renamed_atom14_gt_mask = (
          (1. - alt_naming_is_better[:, :, None]) * batch['atom14_gt_exists']
          + alt_naming_is_better[:, :, None] * batch['atom14_alt_gt_exists'])

  return {
      'alt_naming_is_better': alt_naming_is_better,  # (N)
      'renamed_atom14_gt_positions': renamed_atom14_gt_positions,  # (N, 14, 3)
      'renamed_atom14_gt_exists': renamed_atom14_gt_mask,  # (N, 14)
  }

def frame_aligned_point_error(pred_frames, target_frames, frames_mask, pred_positions, target_positions, positions_mask, pair_mask=None, pos_weight=None, clamp_distance=None, epsilon=1e-8, length_scale=10., unclamped_ratio=0.):
    assert pred_frames[0].shape == target_frames[0].shape
    assert pred_positions.shape == target_positions.shape
    assert list(frames_mask.shape) == list(target_frames[0].shape[:-2])
    assert pred_frames[1].ndim in [3, 4]
    assert pred_positions.ndim in [3, 4]

    batch_size, num_residues = pred_frames[0].shape[:2], pred_frames[0].ndim

    if pred_frames[1].ndim == 4:
        pred_frames = tuple(map(lambda x : rearrange(x, 'b l t ... -> b (l t) ...'), pred_frames))
        target_frames = tuple(map(lambda x : rearrange(x, 'b l t ... -> b (l t) ...'), target_frames))
        frames_mask = rearrange(frames_mask, 'b l t -> b (l t)')

    if pred_positions.ndim == 4:
        pred_positions = rearrange(pred_positions, 'b l t d -> b (l t) d')
        target_positions = rearrange(target_positions, 'b l t d -> b (l t) d')
        positions_mask = rearrange(positions_mask, 'b l t -> b (l t)')

    # pred_frames, (b l 3 3), (b l 3)

    local_pred_pos = r3.rigids_mul_vecs(
            r3.invert_rigids(tuple(map(lambda x : x[:,:,None], pred_frames))),
            pred_positions[:,None])

    local_target_pos = r3.rigids_mul_vecs(
            r3.invert_rigids(tuple(map(lambda x : x[:,:,None], target_frames))),
            target_positions[:,None])

    error_dist = torch.sqrt(
            torch.sum(torch.square(local_pred_pos - local_target_pos), dim=-1) + epsilon)

    if clamp_distance is not None:
        if unclamped_ratio > 0.01:
            error_dist = unclamped_ratio * error_dist + (1. - unclamped_ratio) * torch.clip(error_dist, 0, clamp_distance)
        else:
            error_dist = torch.clip(error_dist, 0, clamp_distance)

    dij_mask = torch.logical_and(frames_mask[:,:,None], positions_mask[:,None])
    if pair_mask is not None:
        dij_mask = torch.logical_and(dij_mask, pair_mask)

    if pos_weight is not None:
        error_dist = error_dist * pos_weight

    error_dist = error_dist * dij_mask

    normalization_factor = torch.sum(dij_mask)

    return torch.sum(error_dist) / length_scale / (epsilon + normalization_factor)

def between_residue_bond_loss(
    pred_atom_positions, pred_atom_mask,
    chain_id, aatype,
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0):

    assert len(pred_atom_positions.shape) == 4
    assert len(pred_atom_mask.shape) == 3
    assert len(chain_id.shape) == 2
    assert len(aatype.shape) == 2

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:, :-1, 1]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:, :-1, 1]         # (N - 1)
    this_c_pos = pred_atom_positions[:, :-1, 2]   # (N - 1, 3)
    this_c_mask = pred_atom_mask[:, :-1, 2]          # (N - 1)
    next_n_pos = pred_atom_positions[:, 1:, 0]    # (N - 1, 3)
    next_n_mask = pred_atom_mask[:, 1:, 0]           # (N - 1)
    next_ca_pos = pred_atom_positions[:, 1:, 1]   # (N - 1, 3)
    next_ca_mask = pred_atom_mask[:, 1:, 1]          # (N - 1)

    has_no_gap_mask = torch.eq(chain_id[:, 1:], chain_id[:, :-1])

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        1e-6 + torch.sum(squared_difference(this_c_pos, next_n_pos), axis=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = torch.eq(aatype[:,1:], residue_constants.resname_to_idx['PRO']).to(dtype=torch.float32)
    gt_length = (
        (1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
        + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = (
        (1. - next_is_proline) *
        residue_constants.between_res_bond_length_stddev_c_n[0] +
        next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = torch.sqrt(1e-6 +
                                     torch.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = F.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev)

    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue) / (torch.sum(mask) + 1e-6)
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    c_ca_unit_vec = l2_normalize(this_ca_pos - this_c_pos)
    c_n_unit_vec = l2_normalize(next_n_pos - this_c_pos)
    n_ca_unit_vec = l2_normalize(next_ca_pos - next_n_pos)

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_cos_angles_ca_c_n[1]
    ca_c_n_cos_angle_error = torch.sqrt(
        1e-6 + torch.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = F.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue) / (torch.sum(mask) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error >
                                    (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        1e-6 + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = F.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue) / (torch.sum(mask) + 1e-6)
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    '''
    per_residue_loss_sum = (c_n_loss_per_residue +
                            ca_c_n_loss_per_residue +
                            c_n_ca_loss_per_residue)
    per_residue_loss_sum = 0.5 * (F.pad(per_residue_loss_sum, (0, 1)) +
            F.pad(per_residue_loss_sum, (1, 0)))
    '''

    '''
    # Compute hard violations.
    violation_mask, _ = torch.max(
        torch.stack([c_n_violation_mask,
                   ca_c_n_violation_mask,
                   c_n_ca_violation_mask], dim=-1), dim=-1)
    violation_mask = torch.maximum(
        F.pad(violation_mask, (0, 1)),
        F.pad(violation_mask, (1, 0)))
    '''
    per_protein_hard_violotion_loss = 0. #torch.mean(torch.sum(violation_mask.to(dtype=torch.float32), dim=1))

    loss_sum = c_n_loss + ca_c_n_loss + c_n_ca_loss

    return loss_sum, {'c_n_loss': c_n_loss,
            'ca_c_n_loss': ca_c_n_loss,
            'c_n_ca_loss': c_n_ca_loss,
            'per_protein_hard_violotion_loss': per_protein_hard_violotion_loss  # shape (N)
           }
