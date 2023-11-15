import numpy as np

from anarci import anarci

from carbonmatrix.data.antibody import antibody_constants

def is_in_framework(domain_type, resid):
    resn, icode = resid

    region_type_idx = antibody_constants.numbering_to_region_type_idx.get((domain_type, resn), None)

    return antibody_constants.region_types[region_type_idx].startswith('FR')

def make_ab_numbering(str_seq, allow):
    # for now, we only support imgt numbering
    results = anarci([('A',str_seq)], scheme='imgt', allow=allow,)

    numbering, alignment_details, hit_tables = results
    if numbering[0] is None:
        return None

    # only return the most significant one
    numbering, query_start, query_end = numbering[0][0]
    query_end += 1
    numbering = [n for n, aatype in numbering if aatype != '-']

    domain_type = alignment_details[0][0]['chain_type']
    # query_start = alignment_details[0][0]['query_start']
    # query_end = alignment_details[0][0]['query_end']

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


# numpy format
def calc_epitope(ab_coords, ab_coord_mask, antigen_coords, antigen_coord_mask, dist_thres=8.0, relax=3):
    # ab_cooords, antigen_coords: (n, 14, 3)
    # ab_coord_mask, antigen_coord_mask

    N = antigen_coords.shape[0]

    # (n, n, 14, 14)
    dist = np.sqrt(np.sum(np.square(antigen_coords[:,None,:,None] - ab_coords[None,:,None,:]), axis=-1))
    mask = antigen_coord_mask[:,None,:,None] * ab_coord_mask[None,:,None,:]

    #dist = dist + (1.0 - mask) * 10e8
    dist = np.where(mask, dist, 10e8)

    min_dist = np.min(dist, axis=(1,2,3))

    # dist2_thres = dist_thres**2
    index = np.argwhere(min_dist < dist_thres)
    index = np.squeeze(index, axis=1)

    if index.shape[0] == 0:
        return None, None

    relax_index = []
    for i in index:
        if min_dist[i] > dist_thres:
            continue
        for k in range(max(0, i - relax), min(i + relax + 1, N)):
            if not relax_index or k > relax_index[-1]:
                relax_index.append(k)

    relax_index = np.array(relax_index)
    relax_dis = min_dist[relax_index]

    return relax_index, relax_dis
