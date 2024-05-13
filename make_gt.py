import numpy as np
import sys
sys.path.append('../')
from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.PDBIO import PDBIO
from carbonmatrix.common import residue_constants
import argparse
import pandas as pd
import os
def make_gt_chain(chain_id, aatypes, coords, coord_mask, residue_ids=None):
    chain = Chain(chain_id)

    serial_number = 1

    def make_residue(resid, aatype, coord, mask, bfactor):
        nonlocal serial_number

        resnum, inscode = resid

        resname = residue_constants.restype_1to3.get(aatype, 'UNK')
        residue = Residue(id=(' ', resnum, inscode), resname=resname, segid='')
        for j, atom_name in enumerate(residue_constants.restype_name_to_atom14_names[resname]):
            if atom_name == '':
                continue
            if not mask[j]:
                continue
            atom = Atom(name=atom_name,
                    coord=coord[j],
                    bfactor=bfactor, occupancy=1, altloc=' ',
                    fullname=str(f'{atom_name:<4s}'),
                    serial_number=serial_number, element=atom_name[:1])
            residue.add(atom)

            serial_number += 1

        return residue

    N = len(aatypes)

    if residue_ids is None:
        residue_ids = [(i + 1, ' ') for i in range(N)]

    for i in range(N):
        bfactor = 0
        if np.sum(coord_mask[i]) > 0:
            chain.add(make_residue(residue_ids[i], aatypes[i], coords[i], coord_mask[i], bfactor))

    return chain


def save_ab_pdb(name, files, output_dir):
    code, heavy_chain, light_chain, antigen_chain = name.split('_')
    model = PDBModel(id=0)
    gt = np.load(files)
    if heavy_chain:
        heavy_seq = gt['heavy_str_seq']
        heavy_coords = gt['heavy_coords']
        heavy_coord_mask = gt['heavy_coord_mask']
        heavy_chain = make_gt_chain(heavy_chain, str(heavy_seq), heavy_coords, heavy_coord_mask)
        model.add(heavy_chain)
    if light_chain:
        light_seq = gt['light_str_seq']
        light_coords = gt['light_coords']
        light_coord_mask = gt['light_coord_mask']
        light_chain = make_gt_chain(light_chain, str(light_seq), light_coords, light_coord_mask)
        model.add(light_chain)

    pdb_ = PDBIO()
    pdb_.set_structure(model)
    pdb_.save(output_dir)

    return

def save_ab_ag_pdb(name, files, output_dir):
    code, heavy_chain, light_chain, antigen_chain = name.split('_')
    model = PDBModel(id=0)
    gt = np.load(files)
    if heavy_chain:
        heavy_seq = gt['heavy_str_seq']
        heavy_coords = gt['heavy_coords']
        heavy_coord_mask = gt['heavy_coord_mask']
        heavy_chain = make_gt_chain(heavy_chain, str(heavy_seq), heavy_coords, heavy_coord_mask)
        model.add(heavy_chain)
    if light_chain:
        light_seq = gt['light_str_seq']
        light_coords = gt['light_coords']
        light_coord_mask = gt['light_coord_mask']
        light_chain = make_gt_chain(light_chain, str(light_seq), light_coords, light_coord_mask)
        model.add(light_chain)
    if antigen_chain:
        antigen_seq = gt['antigen_str_seq']
        antigen_coords = gt['antigen_coords']
        antigen_coord_mask = gt['antigen_coord_mask']
        antigen_chain = make_gt_chain(antigen_chain, str(antigen_seq), antigen_coords, antigen_coord_mask)
        model.add(antigen_chain)

    pdb_ = PDBIO()
    pdb_.set_structure(model)
    pdb_.save(output_dir)

    return


def save_tcr_pdb(name, files, output_dir):
    code, beta_chain, alpha_chain, antigen_chain, mhc_chain = name.split('_')
    model = PDBModel(id=0)
    gt = np.load(files)
    if beta_chain:
        beta_seq = gt['beta_str_seq']
        beta_coords = gt['beta_coords']
        beta_coord_mask = gt['beta_coord_mask']
        beta_chain = make_gt_chain(beta_chain, str(beta_seq), beta_coords, beta_coord_mask)
        model.add(beta_chain)
    if alpha_chain:
        alpha_seq = gt['alpha_str_seq']
        alpha_coords = gt['alpha_coords']
        alpha_coord_mask = gt['alpha_coord_mask']
        alpha_chain = make_gt_chain(alpha_chain, str(alpha_seq), alpha_coords, alpha_coord_mask)
        model.add(alpha_chain)
    pdb_ = PDBIO()
    pdb_.set_structure(model)
    pdb_.save(output_dir)

    return


def save_tcr_pmhc_pdb(name, files, output_dir):
    code, beta_chain, alpha_chain, antigen_chain, mhc_chain = name.split('_')
    model = PDBModel(id=0)
    gt = np.load(files)
    if beta_chain:
        beta_seq = gt['beta_str_seq']
        beta_coords = gt['beta_coords']
        beta_coord_mask = gt['beta_coord_mask']
        beta_chain = make_gt_chain(beta_chain, str(beta_seq), beta_coords, beta_coord_mask)
        model.add(beta_chain)
    if alpha_chain:
        alpha_seq = gt['alpha_str_seq']
        alpha_coords = gt['alpha_coords']
        alpha_coord_mask = gt['alpha_coord_mask']
        alpha_chain = make_gt_chain(alpha_chain, str(alpha_seq), alpha_coords, alpha_coord_mask)
        model.add(alpha_chain)
    if antigen_chain:
        antigen_seq = gt['antigen_str_seq']
        antigen_coords = gt['antigen_coords']
        antigen_coord_mask = gt['antigen_coord_mask']
        antigen_chain = make_gt_chain(antigen_chain, str(antigen_seq), antigen_coords, antigen_coord_mask)
        model.add(antigen_chain)
    if mhc_chain:
        mhc_seq = gt['mhc_str_seq']
        mhc_coords = gt['mhc_coords']
        mhc_coord_mask = gt['mhc_coord_mask']
        mhc_chain = make_gt_chain(mhc_chain, str(mhc_seq), mhc_coords, mhc_coord_mask)
        model.add(mhc_chain)
    pdb_ = PDBIO()
    pdb_.set_structure(model)
    pdb_.save(output_dir)

    return



def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])
    os.makedirs(args.output, exist_ok=True)
    for name in names:
        gt_path = f'{args.gt_dir}/{name}.npz'
        out_ab_path = f'{args.output}/{name}_ab.pdb'
        out_path = f'{args.output}/{name}.pdb'
        try:
            if args.ig == 'ab':
                save_ab_pdb(name, gt_path, out_ab_path)
                save_ab_ag_pdb(name, gt_path, out_path)
            if args.ig == 'tcr':
                save_tcr_pdb(name, gt_path, out_ab_path)
                save_tcr_pmhc_pdb(name, gt_path, out_path)
        except:
            print(f"fail {name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--name_idx', type=str, required=True)
    parser.add_argument('-g', '--gt_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-i', '--ig', type=str, choices=['ab', 'tcr'], default='tcr')
    args = parser.parse_args()

    main(args)


