import numpy as np

import torch
from torch.nn import functional as F
from einops import rearrange

from carbonmatrix.model.utils import l2_normalize

# pylint: disable=bad-whitespace
QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # kr

QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0,-1]]

QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                          [ 1, 0, 0, 0],
                          [ 0, 0, 0, 1],
                          [ 0, 0,-1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                          [ 0, 0, 0,-1],
                          [ 1, 0, 0, 0],
                          [ 0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                          [ 0, 0, 1, 0],
                          [ 0,-1, 0, 0],
                          [ 1, 0, 0, 0]]

QUAT_MULTIPLY_BY_VEC = QUAT_MULTIPLY[:, 1:, :]
# pylint: enable=bad-whitespace

QUAT_TO_ROT_TORCH = torch.tensor(np.reshape(QUAT_TO_ROT, (4, 4, 9)))
QUAT_MULTIPLY_TORCH = torch.tensor(QUAT_MULTIPLY)
QUAT_MULTIPLY_BY_VEC_TORCH = torch.tensor(QUAT_MULTIPLY_BY_VEC)

def make_identity(out_shape, device):
    out_shape = (out_shape) + (3,)
    quaternions = F.pad(torch.zeros(out_shape, device=device), (1, 0), value=1.)
    translations = torch.zeros(out_shape, device = device)

    return quaternions, translations

def quat_to_rot(normalized_quat):
    rot_tensor = torch.sum(
            QUAT_TO_ROT_TORCH.to(normalized_quat.device) *
            normalized_quat[..., :, None, None] *
            normalized_quat[..., None, :, None],
            axis=(-3, -2))
    rot = rearrange(rot_tensor, '... (c d) -> ... c d', c=3, d=3)
    return rot

def quat_multiply_by_vec(quat, vec):
    return torch.sum(
            QUAT_MULTIPLY_BY_VEC_TORCH.to(quat.device) *
            quat[..., :, None, None] *
            vec[..., None, :, None],
            dim=(-3, -2))

def quat_multiply(quat1, quat2):
    assert quat1.shape == quat2.shape
    return torch.sum(
            QUAT_MULTIPLY_TORCH.to(quat1.device) *
            quat1[..., :, None, None] *
            quat2[..., None, :, None],
            dim=(-3, -2))

def quat_precompose_vec(quaternion, vector_quaternion_update):
    assert quaternion.shape[-1] == 4\
            and vector_quaternion_update.shape[-1] == 3\
            and quaternion.shape[:-1] == vector_quaternion_update.shape[:-1]

    new_quaternion = quaternion + quat_multiply_by_vec(quaternion, vector_quaternion_update)
    normalized_quaternion = l2_normalize(new_quaternion)

    return normalized_quaternion

def quat_invert(quaternion: torch.Tensor) -> torch.Tensor:
    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = torch.lt(torch.abs(angles.detach()), eps)
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    minus_sign = torch.lt(quaternions.detach()[...,0], 0.)
    quaternions[minus_sign] = -1.0 * quaternions[minus_sign]

    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = torch.lt(torch.abs(angles.detach()), eps)
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles
