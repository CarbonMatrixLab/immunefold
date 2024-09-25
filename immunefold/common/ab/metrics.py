import numpy as np

from collections import OrderedDict

from Bio.PDB.Chain import Chain as PDBChain
from Bio.PDB.Residue import Residue

from immunefold.common.metrics import Kabsch

def get_antibody_regions(N, struc2seq, chain_id, schema='imgt'):
    assert chain_id in 'HL'
    assert schema in ['chothia', 'imgt']
    cdr_def_chothia = {
            'H': {
                'fr1' : (1,  25),
                'cdr1': (26, 32),
                'fr2' : (33, 51),
                'cdr2': (52, 56),
                'fr3' : (57, 94),
                'cdr3': (95, 102),
                'fr4' : (103,113),
            },
            'L': {
                'fr1' : (1,  23),
                'cdr1': (24, 34),
                'fr2' : (35, 49),
                'cdr2': (50, 56),
                'fr3' : (57, 88),
                'cdr3': (89, 97),
                'fr4' : (98, 109),
                }
    }

    cdr_def_imgt = {
            'H': {
                'fr1' : (1,  26),
                'cdr1': (27, 38),
                'fr2' : (39, 55),
                'cdr2': (56, 65),
                'fr3' : (66, 104),
                'cdr3': (105,117),
                'fr4' : (118,128),
            },
            'L': {
                'fr1' : (1,  26),
                'cdr1': (27, 38),
                'fr2' : (39, 55),
                'cdr2': (56, 65),
                'fr3' : (66, 104),
                'cdr3': (105,117),
                'fr4' : (118,128),
            },
    }

    cdr_def = cdr_def_imgt if schema == 'imgt' else cdr_def_chothia

    range_dict = cdr_def[chain_id]

    _schema = {'fr1':0,'cdr1':1,'fr2':2,'cdr2':3,'fr3':4,'cdr3':5,'fr4':6}

    def _get_region(i):
        r = None
        for k, v in range_dict.items():
            if i >= v[0] and i <= v[1]:
                r = k
                break
        if r is None:
            return -1
        return 7 * int(chain_id == 'L') + _schema[r]

    region_def = np.full((N,),-1)

    for (hetflag, resseq, icode), v in struc2seq.items():
        region_def[v] = _get_region(int(resseq))

    return region_def

def get_antibody_regions_seq(imgt_numbering, chain_id):
    cdr_def_imgt = {
            'H': {
                'fr1' : (1,  26),
                'cdr1': (27, 38),
                'fr2' : (39, 55),
                'cdr2': (56, 65),
                'fr3' : (66, 104),
                'cdr3': (105,117),
                'fr4' : (118,128),
            },
            'L': {
                'fr1' : (1,  26),
                'cdr1': (27, 38),
                'fr2' : (39, 55),
                'cdr2': (56, 65),
                'fr3' : (66, 104),
                'cdr3': (105,117),
                'fr4' : (118,128),
            },
    }

    cdr_def = cdr_def_imgt

    range_dict = cdr_def[chain_id]

    _schema = {'fr1':0,'cdr1':1,'fr2':2,'cdr2':3,'fr3':4,'cdr3':5,'fr4':6}

    def _get_region(i):
        r = None
        for k, v in range_dict.items():
            if i >= v[0] and i <= v[1]:
                r = k
                break
        if r is None:
            return -1
        return 7 * int(chain_id == 'L') + _schema[r]

    N = len(imgt_numbering)
    region_def = np.full((N,),-1)

    for i, (_, resseq, icode) in enumerate(imgt_numbering):
        region_def[i] = _get_region(resseq)

    return region_def

def calc_ab_metrics(gt_ab_coords, pred_ab_coords, ab_coord_mask, cdr_def, remove_middle_residues=False, mode='unbound', nano=False):
    ab_mask = ab_coord_mask
    # try:
    gt_ab_coords, pred_ab_coords = gt_ab_coords[ab_mask,:], pred_ab_coords[ab_mask, :]
    if nano:
        framework_indices = np.any(np.stack([cdr_def == k for k in [0,2,4,6]], axis=0), axis=0)
    else:
        framework_indices = np.any(np.stack([cdr_def == k for k in [0,2,4,6,7,9,11,13]], axis=0), axis=0)
    # gt_coord, pred_coord = gt_coord[mask,:], pred_coord[mask, :]
    # except:
        # import pdb
        # pdb.set_trace()
    # gt_aligned, pred_aligned = Kabsch(
    #         np.transpose(gt_coord,[1,0]),
    #         np.transpose(pred_coord, [1, 0]))
    gt_aligned, pred_aligned = Kabsch(
            np.transpose(gt_ab_coords,[1,0]),
            np.transpose(pred_ab_coords, [1, 0]))
    # import pdb
    # pdb.set_trace()
    def _calc_rmsd(A, B):
        return np.sqrt(np.mean(np.sum(np.square(A-B), axis=0)))

    full_rmsd = _calc_rmsd(gt_aligned, pred_aligned)

    ret = OrderedDict()
    ret.update({'full_len' : gt_aligned.shape[1]})
    ret.update({'full_rmsd':full_rmsd})

    _schema = {'fr1':0,'cdr1':1,'fr2':2,'cdr2':3,'fr3':4,'cdr3':5,'fr4':6}
    cdr_idx = {v : 'heavy_' + k for k, v in _schema.items()}
    if nano: 
        pass
    else: 
        cdr_idx.update({v + 7 : 'light_' + k for k, v in _schema.items()})
    if mode == 'bound':
        cdr_idx.update(
            {20: 'antigen'}
        )
    def _evaluate_region(indices, v):
        seq_len = np.sum(indices)
        struc_len = np.sum(indices[ab_mask])
        coverage = struc_len / seq_len
        indices = indices[ab_mask]
        gt, pred = gt_aligned[:, indices], pred_aligned[:, indices]
        rmsd = _calc_rmsd(gt, pred)
        ret.update({v + '_len' : seq_len})
        ret.update({v + '_rmsd':rmsd})
        ret.update({v + '_coverage':coverage})

        return

    for k, v in cdr_idx.items():
        indices = (cdr_def == k)
        if remove_middle_residues and v in ['heavy_cdr3',]:
            indices = np.logical_and(indices, np.concatenate([indices, np.zeros(2,)])[2:])
            indices = np.logical_and(indices, np.concatenate([np.zeros(3,), indices])[:-3])

        _evaluate_region(indices, v)

    # all framework regions
    indices = np.any(np.stack([cdr_def == k for k in [0,2,4,6,7,9,11,13]], axis=0), axis=0)
    _evaluate_region(indices, 'all_frs')

    # all cdr regions except cdr3
    indices = np.any(np.stack([cdr_def == k for k in [1,3,8,10,12]], axis=0), axis=0)
    _evaluate_region(indices, 'other cdrs')

    return ret

def renum_chain_imgt(orig_chain, struc2seq, imgt_numbering):
    chain = PDBChain(orig_chain.id)
    for residue in orig_chain:
        if residue.id in struc2seq:
            idx = struc2seq[residue.id]
            new_residue = Residue(id=imgt_numbering[idx], resname=residue.resname, segid=residue.segid)
            for atom in residue:
                atom.detach_parent()
                new_residue.add(atom)
            chain.add(new_residue)
    return chain
