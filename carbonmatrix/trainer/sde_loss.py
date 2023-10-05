from einops import rearrange
import torch

from carbonmatrix.trainer.loss_factory import registry_loss
from carbonmatrix.sde.se3_diffuser import SE3Diffuser
from carbonmatrix.model import quat_affine

@registry_loss
def score_loss(batch, value, config):
    c = config

    value = value['heads']['structure_module']
    _, pred_trans = value['traj'][-1]
    
    gt_trans, trans_mask = batch['atom14_gt_positions'][...,1,:], batch['atom14_gt_exists'][...,1]
    trans_loss = compute_trans_loss(gt_trans, pred_trans, trans_mask, c.trans_clamp_distance, c.trans_length_scale)
    print('batch t', batch['t'])
    
    diffuser = SE3Diffuser.get(config) 
    
    gt_rot_score = batch['rot_score']
    rot_score_scaling = batch['rot_score_scaling']
    
    delta_quat = value['delta_quat']
    quat_inv0_t = quat_affine.quat_invert(delta_quat)
    pred_rot_score = diffuser.calc_rot_score(quat_inv0_t, batch['t'])
    rot_mask = batch['rigidgroups_gt_exists'][...,0]

    rot_loss, rot_loss_items = compute_rot_loss(gt_rot_score, pred_rot_score,
            rot_score_scaling, rot_mask,
            t = batch['t'],
            rot_angle_loss_weight = c.rot_angle_loss_weight,
            rot_angle_loss_t_threshold = c.rot_angle_loss_t_threshold)

    score_loss = c.trans_loss_weight * trans_loss + rot_loss
    
    return dict(loss = score_loss,
            rot_score_loss = rot_loss,
            trans_score_loss = trans_loss,
            **rot_loss_items,
            )

def compute_trans_loss(gt_trans, pred_trans, trans_mask, clamp_distance, length_scale):
    # Translation x0 loss
    error = torch.sum(torch.square(pred_trans - gt_trans), dim=-1)
    error = torch.clip(error, 0., clamp_distance**2)
    square_length_scale = length_scale**2
    error = torch.sum(error * trans_mask) / (torch.sum(trans_mask) + 1e-10) / square_length_scale
    return error

def compute_rot_loss(gt_rot_score, pred_rot_score, rot_score_scaling, rot_mask,
        t,
        rot_angle_loss_weight, rot_angle_loss_t_threshold):
    # Rotation loss
    gt_rot_angle = torch.linalg.norm(gt_rot_score, dim=-1, keepdim=True)
    gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-10)

    pred_rot_angle = torch.linalg.norm(pred_rot_score, dim=-1, keepdim=True)
    pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-10)

    # Separate loss on the axis
    axis_loss = torch.sum(torch.square(gt_rot_axis - pred_rot_axis), dim=-1)
    axis_loss = torch.sum(axis_loss * rot_mask) / (torch.sum(rot_mask) + 1e-10)

    # Separate loss on the angle
    # (b, l)
    angle_loss = torch.square(rearrange(gt_rot_angle - pred_rot_angle, '... () -> ...'))

    angle_mask = rot_mask * rearrange(torch.gt(t, rot_angle_loss_t_threshold), 'b -> b ()')
    angle_loss = torch.sum(angle_mask * angle_loss / rearrange(torch.square(rot_score_scaling), 'b -> b ()')) / (torch.sum(angle_mask) + 1e-10)

    rot_loss = rot_angle_loss_weight * angle_loss + axis_loss
    
    rot_loss_items = dict(
            rot_axis_loss = axis_loss,
            rot_angle_loss = angle_loss
            )
    
    return rot_loss, rot_loss_items
