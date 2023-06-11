import os
import argparse
from igfold import IgFoldRunner


def main(args):
    igfold = IgFoldRunner()

    def _predict(seqs, pdb_file):
        igfold.fold(
                pdb_file, # Output PDB file
                sequences=seqs, # Antibody sequences
                do_refine=True, # Refine the antibody structure with PyRosetta
                use_openmm=True, # Use OpenMM for refinement
                do_renum=False, # Renumber predicted antibody structure (Chothia)
        )
    
    with open(args.name_idx) as f:
        for line in f:
            name = line.strip()
            fasta_file = os.path.join(args.fasta_dir, name + '.fasta')

            with open(fasta_file) as f2:
                lines = f2.readlines()
            
            H = lines[1].strip()
            L = lines[3].strip()
            
            seqs = {'H':H, 'L':L}
            pdb_file = os.path.join(args.pdb_dir, name + '.pdb')

            try:
                _predict(seqs, pdb_file)
            except:
                print('error', name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--fasta_dir', type=str, required=True)
    parser.add_argument('--pdb_dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)
