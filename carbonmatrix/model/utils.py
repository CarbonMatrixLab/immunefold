import torch
from torch.nn import functional as F
from einops import rearrange, parse_shape
import torch
from einops import rearrange

def to_centroid(points, keepdim=True, mask=None):
    '''
    points: (b, n, 3)
    mask: (b, n)
    '''
    if mask is None:
        return torch.mean(points, dim=1, keepdim=keepdim)

    assert(points.ndim == mask.ndim + 1)

    return torch.sum(points * mask[...,None], dim=-2, keepdim=keepdim) / (1e-10 + torch.sum(mask, dim=-1, keepdim=keepdim)[...,None])

@torch.no_grad()
def batch_kabsch(other_points, ref_points, mask, needs_to_be_centered=True):
    '''
    other_points: (b, n, 3)
    ref_points: (b, n, 3)
    mask: (b, n)
    does not support batch ot yet
    '''
    assert(ref_points.ndim == other_points.ndim)
    assert(ref_points.ndim == mask.ndim + 1)

    ref_points_mean = torch.mean(ref_points * mask[...,None], dim=1)
    other_points_mean = torch.mean(other_points * mask[...,None], dim=1)
    

    if needs_to_be_centered:
        ref_points = ref_points- to_centroid(ref_points, mask=mask)
        other_points = other_points - to_centroid(other_points, mask=mask)
    

    other_points = other_points * mask[...,None]
    ref_points = ref_points * mask[...,None]

    # (b, 3, 3)
    covariance = torch.einsum('... n a, ... n b -> ... a b', other_points, ref_points)

    # U, (b, 3, 3); S, (b, 3); Vt: (b, 3, 3)
    U, S, Vt = torch.linalg.svd(covariance)

    # (b,)
    sign_correction = torch.sign(torch.linalg.det(torch.matmul(U, Vt)))

    U[...,:,-1] = U[...,:,-1] * sign_correction[...,None]

    # (b, 3, 3)
    rot = torch.matmul(U, Vt)
    trans = ref_points_mean - torch.einsum('a b, a b n -> a n', other_points_mean, rot)
    # trans = ref_points_mean - torch.matmul(other_points_mean, rot)
    # import pdb
    # pdb.set_trace()
    # (b, n, 3)
    # other_points_aligned = torch.matmul(other_points, rot)

    return rot, trans

def l2_normalize(v, dim=-1, epsilon=1e-8):
    norms = torch.sqrt(torch.sum(torch.square(v), dim=dim, keepdims=True)) + epsilon
    return v / norms

def squared_difference(x, y):
    return torch.square(x-y)

def batched_select(params, indices, dim=None, batch_dims=0):
    params_shape, indices_shape = list(params.shape), list(indices.shape)
    assert params_shape[:batch_dims] == indices_shape[:batch_dims]

    def _permute(dim, dim1, dim2):
        permute = []
        for i in range(dim):
            if i == dim1:
                permute.append(dim2)
            elif i == dim2:
                permute.append(dim1)
            else:
                permute.append(i)
        return permute

    if dim is not None and dim != batch_dims:
        params_permute = _permute(len(params_shape), dim1=batch_dims, dim2=dim)
        indices_permute = _permute(len(indices_shape), dim1=batch_dims, dim2=dim)
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)
        params_shape, indices_shape = list(params.shape), list(indices.shape)

    params, indices = torch.reshape(params, params_shape[:batch_dims+1] + [-1]), torch.reshape(indices, list(indices_shape[:batch_dims]) + [-1, 1])

    # indices = torch.tile(indices, params.shape[-1:])
    indices = indices.repeat([1] * (params.ndim - 1) + [params.shape[-1]])

    batch_params = torch.gather(params, batch_dims, indices.to(dtype=torch.int64))

    output_shape = params_shape[:batch_dims] + indices_shape[batch_dims:] + params_shape[batch_dims+1:]

    if dim is not None and dim != batch_dims:
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)

    return torch.reshape(batch_params, output_shape)
