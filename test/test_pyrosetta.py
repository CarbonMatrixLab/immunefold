import sys

import pyrosetta
from pyrosetta.rosetta.protocols.antibody import CDRNameEnum

init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
pyrosetta.init(init_string)

def test(path):
    pose = pyrosetta.pose_from_pdb(path)
    #sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector('L')
    #pose = pose[sel.apply(pose)]

    #print(pose)
    #for a in pose:
    #    print(a)
    info = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose)
    print(info.get_CDR_start_PDB_num(CDRNameEnum.h1), info.get_CDR_end_PDB_num(CDRNameEnum.h1))
    print(info.get_CDR_start_PDB_num(CDRNameEnum.h2), info.get_CDR_end_PDB_num(CDRNameEnum.h2))
    print(info.get_CDR_start_PDB_num(CDRNameEnum.h3), info.get_CDR_end_PDB_num(CDRNameEnum.h3))
    

    #CDRs
    cdrs = info.get_all_cdrs()
    for c in cdrs:
        print(c)

    #numbering
    numbering = info.get_antibody_numbering_info()
    print(numbering.cdr_definition)
    print(numbering.cdr_numbering)
    print(numbering.numbering_scheme)
    print(region)
    


if __name__ == '__main__':
    test(sys.argv[1])
