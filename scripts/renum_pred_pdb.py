import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd

from abfold.preprocess.parser import parse_pdb
from Bio.PDB.Chain import Chain as PDBChain
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB import PDBIO

def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    return lines[1].strip(), lines[3].strip()

def _update_resnum(line, new_resnum):
    return line[:22] + f'{str(new_resnum):>4s}' + line[26:]
def _update_chain(line, new_chain):
    return line[:21] + new_chain + line[22:]


def make_ig_one(name, seq_file, pred_file, output_file):
    str_heavy_seq, str_light_seq = read_fasta(seq_file)
    
    light_len = len(str_light_seq)
    sep_len = 50

    with open(pred_file) as f, open(output_file, 'w') as fw:
        ter_flag = True
        for line in f:
            if line.startswith('ATOM') or line.startswith('TER'):
                resnum = int(line[22:26]) - 1
                if resnum < light_len:
                    new_resnum = resnum
                    new_chain = 'L'
                elif resnum < light_len + sep_len:
                    if resnum + 1 == light_len + sep_len and ter_flag:
                        fw.write('TER\n')
                        ter_flag = False
                    continue
                else:
                    new_resnum = resnum - light_len - sep_len
                    new_chain = 'H'
                new_line = _update_resnum(line.strip(), new_resnum)
                new_line = _update_chain(new_line, new_chain)
    
                fw.write(new_line + '\n')
            else:
                new_line = line.strip()
                fw.write(new_line + '\n')

def make_ig_one2(name, seq_file, pred_file, output_file):
    str_heavy_seq, str_light_seq = read_fasta(seq_file)
    
    light_len = len(str_light_seq)
    sep_len = 50
    struc = parse_pdb(pred_file)
    chain = list(struc.get_chains())[0]
    
    heavy_chain = PDBChain('H')
    light_chain = PDBChain('L')
    def _add_new_residue(residue, chain_id):
        residue.detach_parent()
        if chain_id == 'H':
            heavy_chain.add(residue)
        if chain_id == 'L':
            light_chain.add(residue)

    for idx, residue in enumerate(chain):
        if idx < light_len:
            _add_new_residue(residue, 'L')
        elif idx >= light_len + sep_len:
            _add_new_residue(residue, 'H')
    
    model = PDBModel('0')
    pdb_io = PDBIO()
    model.add(heavy_chain)
    model.add(light_chain)

    pdb_io.set_structure(model)
    pdb_io.save(output_file)

def renum_ig(names, args):
    for i, n in enumerate(names):
        seq_file = os.path.join(args.fasta_dir, n + '.fasta')
        pred_file = os.path.join(args.pred_dir, n + '.pdb')
        output_file = os.path.join(args.output_dir, n + '.pdb')
        if os.path.exists(seq_file) and os.path.exists(pred_file):
            print(pred_file)
            print(output_file)
            make_ig_one2(n, seq_file, pred_file, output_file)
            
def make_general_one(name, gt_npz_file, gt_pdb_file, pred_file, output_file):
    
    gt_struc = np.loadd(gt_npz_file)
    gt_seq = str(gt_struc['str_seq'])
    gt_mask = gt_struc['atom14_gt_exists'][:,1]
    
    parser = PDBParser()
    gt_chain = parser.get_structure(name, pdb_file)[0].get_chains[0]
    assert len(gt_chain) == gt_mask.shape[0]
    
def renum_general(names, args):
    for i, n in enumerate(names):
        gt_npz_file = os.path.join(args.gt_dir, n + '.npz')
        gt_pdb_file = os.path.join(args.gt_dir, n + '.pdb')
        pred_file = os.path.join(args.pred_dir, n + '.pdb')
        output_file = os.path.join(args.output_dir, n + '.pdb')
        if os.path.exists(gt_npz_file) and os.path.exists(pred_file):
            print(pred_file)
            print(output_file)
            make_general_one(n, gt_npz_file, gt_pdb_file, pred_file, output_file)
        break

def main(args):
    names = list(pd.read_csv(args.name_idx, names=['name'], header=None)['name'])

    renum_ig(names, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['ig', 'general'], required=True)
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--fasta_dir', type=str, required=True, help='fasta dir')
    parser.add_argument('--pred_dir', type=str, required=True, help='predict pdb dir')
    parser.add_argument('--output_dir', type=str, required=True, help='output dir, default=\'.\'')
    args = parser.parse_args()
    
    main(args)
