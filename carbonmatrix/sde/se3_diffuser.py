import logging

import numpy as np
import torch

from carbonmatrix.model import quat_affine
from carbonmatrix.sde import so3_diffuser, r3_diffuser

def _extract_trans_rots(rigid):
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] +(3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot

def _assemble_rigid(rotvec, trans):
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = Rotation.from_rotvec(rotvec).as_matrix().reshape(
        rotvec_shape[:-1] + (3, 3))
    return ru.Rigid(
            rots=ru.Rotation(
                rot_mats=torch.Tensor(rotmat)),
            trans=torch.tensor(trans))

diffuser_obj_dict = {}

class SE3Diffuser(object):
    def __init__(self, se3_conf):
        if isinstance(se3_conf, dict):
            se3_conf = ConfigDict(se3_conf)
        
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)

    @staticmethod
    def get(se3_conf):
        global diffuser_obj_dict
        
        name = 'diffuser'
        
        if name not in diffuser_obj_dict:
            diffuser_obj_dict[name] = SE3Diffuser(se3_conf)
        
        return diffuser_obj_dict[name]

    def forward_marginal(self, rigids_0, t, diffuse_mask=None):
        (rot_0, trans_0) = rigids_0

        rot_t, rot_score = self._so3_diffuser.forward_marginal(rot_0, t)
        rot_score_scaling = self._so3_diffuser.score_scaling(t)

        trans_t, trans_score = self._r3_diffuser.forward_marginal(trans_0, t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)

        if diffuse_mask is not None:
            # diffuse_mask = torch.tensor(diffuse_mask).to(rot_t.device)
            rot_t = self._apply_mask(
                rot_t, rot_0, diffuse_mask[..., None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score,
                np.zeros_like(trans_score),
                diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score,
                np.zeros_like(rot_score),
                diffuse_mask[..., None])
        
        rigids_t = (rot_t, trans_t)
        
        return {
            'rigids_t': rigids_t,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return self._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    def calc_rot_score(self, quat, t):
        axis_angle = quat_affine.quaternion_to_axis_angle(quat)

        return self._so3_diffuser.score(axis_angle, t)

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def score(
            self,
            rigid_0,
            rigid_t,
            t: float):
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_score = np.zeros_like(rot_0)
        else:
            rot_score = self._so3_diffuser.score(
                rot_t, t)

        if not self._diffuse_trans:
            trans_score = np.zeros_like(tran_0)
        else:
            trans_score = self._r3_diffuser.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
            self,
            rigid_t,
            rot_score: np.ndarray,
            trans_score: np.ndarray,
            t: float,
            dt: float,
            diffuse_mask: np.ndarray = None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
                )
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                center=center,
                noise_scale=noise_scale
                )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(
                rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid(rot_t_1, trans_t_1)

    def sample_ref(
            self,
            n_samples: int,
            impute=None,
            diffuse_mask: np.ndarray=None,
            as_tensor_7: bool=False
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))
            trans_impute = self._r3_diffuser._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_rot) and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_trans) and impute is None:
            raise ValueError('Must provide imputation values.')

        if self._diffuse_rot:
            rot_ref = self._so3_diffuser.sample_ref(
                n_samples=n_samples)
        else:
            rot_ref = rot_impute

        if self._diffuse_trans:
            trans_ref = self._r3_diffuser.sample_ref(
                n_samples=n_samples
            )
        else:
            trans_ref = trans_impute

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])
        trans_ref = self._r3_diffuser._unscale(trans_ref)
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {'rigids_t': rigids_t}
