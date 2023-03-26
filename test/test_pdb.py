from abfold.preprocess.parser import parse_pdb, pdb_get_coords
from Bio.SeqIO.FastaIO import SimpleFastaParser as FastaParser

fasta_seq = {}
with open("./examples/data/12e8_trunc.fasta") as handle:
    fasta_seq_dict={k.split(':')[-1]:v for k,v in FastaParser(handle)}
print(fasta_seq_dict)

structure = parse_pdb('./examples/data/12e8_trunc.pdb')
heavy_coord, heavy_coord_mask = pdb_get_coords(structure['H'], fasta_seq_dict['H'])
