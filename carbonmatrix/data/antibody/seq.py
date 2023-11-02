import numpy as np

from anarci import anarci

from carbonmatrix.data.antibody import antibody_constants

def make_ab_numbering(str_seq, allow):
    # for now, we only support imgt numbering
    results = anarci([('A',str_seq)], scheme='imgt', allow=allow,)

    numbering, alignment_details, hit_tables = results
    if numbering[0] is None:
        return None
    
    # only return the most significant one
    numbering, _, _ = numbering[0][0] 
    print('xxxx', numbering)
    numbering = [n for n, aatype in numbering if aatype != '-']
    
    domain_type = alignment_details[0][0]['chain_type']
    query_start = alignment_details[0][0]['query_start']
    query_end = alignment_details[0][0]['query_end']
    
    assert(query_end - query_start == len(numbering))

    ret = dict(
        query_start=query_start,
        query_end=query_end,
        numbering=numbering,
        region_index = make_ab_region(domain_type, numbering)
        )
    
    return ret
        
def make_ab_region(domain_type, numbering):
    region_index = [antibody_constants.numbering_to_region_type_idx[(domain_type, resnum)] for resnum, inscode in numbering]

    return np.array(region_index)