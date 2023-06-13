import numpy as np


from Bio.PDB.internal_coords import IC_Chain as PDBChain
from Bio.PDB.vectors import calc_angle, calc_dihedral
from Bio.PDB.vectors import Vector

from abfold.preprocess.parser import parse_pdb
from abfold.common import residue_constants

def rigids_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane):
    #v1 = point_on_x_axis - origin
    v1 = origin - point_on_neg_x_axis
    #if reverse:
    #    v1 = -v1
    v2 = point_on_xy_plane - origin
    e1 = v1 / np.linalg.norm(v1)

    c =  np.sum(e1 * v2)
    u2 = v2 - c*e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)

    rots = np.stack([e1, e2, e3], axis=1)

    return rots, origin

def rigids_from_2_vecss(v1, v2, origin):
    e1 = v1 / np.linalg.norm(v1)

    c =  np.sum(e1 * v2)
    u2 = v2 - c*e1
    e2 = u2 / np.linalg.norm(u2)
    e3 = np.cross(e1, e2)

    rots = np.stack([e1, e2, e3], axis=1)

    return rots, origin

def frame_from_angle(angle):
    s, c = np.sin(angle), np.cos(angle)
    return np.array(
            [[1, 0, 0],
             [0, c, -s],
             [0, s, c]])


def rigids_from_4x4(x):
    rot = x[...,:3,:3]
    tran = x[...,:3,3]
    return rot, tran

def rigids_mul(r1, r2):
    rot1, tran1 = r1
    rot2, tran2 = r2
    
    rot = np.matmul(rot1, rot2)
    tran = np.matmul(rot1, tran2[:,None])[:,0] + tran1

    return (rot, tran)

def rigid_mul_vec(r, v):
    rot, tran = r
    return np.matmul(rot, v[:,None])[:,0] + tran

def invert_rigid(r):
    rot, tran = r
    inv_rot = np.transpose(rot, [1,0])
    inv_tran = -np.matmul(inv_rot, tran[:,None])[:,0]

    return (inv_rot, inv_tran)

def angle_to_pi(x):
    return x / 180.0 * np.pi

def test_residue(r, prev_r, next_r):
    ric = r.internal_coord
    #phi, psi, omega = ric.get_angle('phi'), ric.get_angle('psi'), ric.get_angle('omega')
    #phi, psi, omega = angle_to_pi(phi), angle_to_pi(psi), angle_to_pi(omega)
    
    print(r)
    print('backbone frame')
    rot, tran = rigids_from_3_points(r['C'].coord, r['CA'].coord, r['N'].coord)
    print(rot, tran)
    rot_x = np.array([[-1, 0., 0.],
                       [0, 1, 0,],
                       [0, 0, 1]])
    tran_x = np.array([0, 0., 0.])
    print(rot.shape, tran.shape, rot_x.shape, tran_x.shape)
    print('CA', rigid_mul_vec(invert_rigid((rot, tran)), r['CA'].coord))
    print('N', rigid_mul_vec(invert_rigid(rigids_mul((rot, tran), (rot_x, tran_x))), r['N'].coord))
    print('C', rigid_mul_vec(invert_rigid(rigids_mul((rot, tran), (rot_x, tran_x))), r['C'].coord))
    
    return

    # calculate the angles from the scratch
    phi = calc_dihedral(Vector(prev_r['C'].coord), Vector(r['N'].coord), Vector(r['CA'].coord), Vector(r['C'].coord))
    psi = calc_dihedral(Vector(r['N'].coord), Vector(r['CA'].coord), Vector(r['C'].coord), Vector(r['O'].coord))
    omega = calc_dihedral(Vector(prev_r['CA'].coord), Vector(prev_r['C'].coord), Vector(r['N'].coord), Vector(r['CA'].coord))

    print('check phi',  phi, np.sin(phi), np.cos(phi))
    print('check psi',  psi, np.sin(psi), np.cos(psi))
    print('check pre_omega', omega)

    # CA-C-O frame
    o_angle = calc_angle(Vector(r['CA'].coord), Vector(r['C'].coord), Vector(r['O'].coord))
    print('CA-C-O angle', o_angle)

    print('backbone frame')
    bb_rot, bb_tran = rigids_from_3_points(r['C'].coord, r['CA'].coord, r['N'].coord)
    print(bb_rot)
    print(bb_tran)
    
    print('recheck the psi frame')
    print('psi default frame')
    default_psi_frame = residue_constants.restype_rigid_group_default_frame[:,3]
    psi_rot, psi_tran = rigids_from_4x4(default_psi_frame[0])
    print(psi_rot)
    print(psi_tran)

    print('psi angle frame')
    angle_rot = frame_from_angle(psi)
    print(angle_rot)
    
    print('default and angle frame compose')
    atom_rot = np.matmul(psi_rot, angle_rot)
    atom_tran = psi_tran
    print(atom_rot)
    print(atom_tran)

    print('backbone frame compose')
    g_frame, g_tran = rigids_mul((bb_rot, bb_tran), (atom_rot, atom_tran))
    print(g_frame)
    print(g_tran)

    print('verified psi frame')
    v_psi_rot, v_psi_tran = rigids_from_3_points(r['CA'].coord, r['C'].coord, r['O'].coord, reverse=True)
    print(v_psi_rot)

    print('groud truth O atom')
    print(r['O'].coord)

    print('restored O atom')
    default_o_pos = np.array([0.626, 1.062, 0.000]) 
    o_pos = rigid_mul_vec((g_frame, g_tran), default_o_pos)
    print(o_pos)

    print('checked psi')
    psi = calc_dihedral(Vector(r['N'].coord), Vector(r['CA'].coord), Vector(r['C'].coord), Vector(o_pos))
    print(psi, np.sin(psi), np.cos(psi))

    print('recheck the phi frame')
    print('phi default frame')
    default_phi_frame = residue_constants.restype_rigid_group_default_frame[:,2]
    phi_rot, phi_tran = rigids_from_4x4(default_phi_frame[0])
    print(phi_rot)
    print(phi_tran)

    print('phi angle frame')
    angle_rot = frame_from_angle(phi)
    print(angle_rot)
    
    print('default and angle frame compose')
    atom_rot = np.matmul(phi_rot, angle_rot)
    atom_tran = phi_tran
    print(atom_rot)
    print(atom_tran)

    print('backbone frame compose')
    g_frame, g_tran = rigids_mul((bb_rot, bb_tran), (atom_rot, atom_tran))
    print(g_frame)
    print(g_tran)

    print('verified phi frame')
    v_phi_rot, v_phi_tran = rigids_from_3_points(r['CA'].coord, r['N'].coord, prev_r['C'].coord, reverse=False)
    print(v_phi_rot)
    print(v_phi_tran)

def test_one(path, center_pos):
    struc_model = parse_pdb(path, model = 0)

    ic_chain = PDBChain(struc_model['A'])
    ic_chain.atom_to_internal_coordinates()
    chain = ic_chain.chain

    r, prev_r, next_r = chain[(' ',center_pos,' ')], chain[(' ',center_pos-1,' ')], chain[(' ',center_pos+1,' ')]
    test_residue(r, prev_r, next_r)

def _calc_dihedral(v1, v2, v3, v4):
    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3
    u = ab**cb
    v = db**cb
    w = u**v
    angle = u.angle(v)


if __name__ == '__main__':
    test_one('./examples/1ctf.gt.pdb', center_pos=56)
