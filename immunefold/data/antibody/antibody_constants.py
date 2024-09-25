domain_types = ['H', 'L', 'K']
region_types = ['FR-H1', 'CDR-H1', 'FR-H2', 'CDR-H2', 'FR-H3', 'CDR-H3', 'FR-H4',
                'FR-L1', 'CDR-L1', 'FR-L2', 'CDR-L2', 'FR-L3', 'CDR-L3', 'FR-L4']

domain_type_to_region_type = {
    'H': ['FR-H1', 'CDR-H1', 'FR-H2', 'CDR-H2', 'FR-H3', 'CDR-H3', 'FR-H4'],
    'L': ['FR-L1', 'CDR-L1', 'FR-L2', 'CDR-L2', 'FR-L3', 'CDR-L3', 'FR-L4'],
    'K': ['FR-L1', 'CDR-L1', 'FR-L2', 'CDR-L2', 'FR-L3', 'CDR-L3', 'FR-L4'],
}
region_type_to_domain_type = {}
for domain, regions in domain_type_to_region_type.items():
    region_type_to_domain_type.update({region:domain for region in regions})

region_type_order = {region_type: i for i, region_type in enumerate(region_types)}
region_type_num = len(region_types)
unk_region_type_index = region_type_num

region_types_with_x = region_types + ['UNK']
region_type_order_with_x = {region_type: i for i, region_type in enumerate(region_types_with_x)}

# all regions are left-closed and right-open.
imgt_region_def = {
# heavy
    'FR-H1' : (1,  27),
    'CDR-H1': (27, 39),
    'FR-H2' : (39, 56),
    'CDR-H2': (56, 66),
    'FR-H3' : (66, 105),
    'CDR-H3': (105,118),
    'FR-H4' : (118,129),
# light
    'FR-L1' : (1,  27),
    'CDR-L1': (27, 39),
    'FR-L2' : (39, 56),
    'CDR-L2': (56, 66),
    'FR-L3' : (66, 105),
    'CDR-L3': (105,118),
    'FR-L4' : (118,129),
}

def __make_numbering_to_region_type_idx():
    ret = {}
    for domain_type, region_types in domain_type_to_region_type.items():
        for region_type in region_types:
            start_pos, end_pos = imgt_region_def[region_type]
            ret.update({(domain_type, x) : region_type_order[region_type] for x in range(start_pos, end_pos)})
    return ret

numbering_to_region_type_idx = __make_numbering_to_region_type_idx() 