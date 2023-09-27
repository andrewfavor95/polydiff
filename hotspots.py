import torch
import numpy as np

def fix_indices(idx):
    new_idx = []
    current_offset = 0
    for i, iplus1 in zip(idx[:-1], idx[1:]):
        if iplus1 - i >= 200:
            current_offset += 200
        new_idx.append(i - current_offset)
    new_idx.append(idx[-1] - current_offset)
    return np.array(new_idx)

def make_hotspot_vector(indep, hotspot_dfs_dict, hotspot_id):
    hotspot = torch.zeros(indep.length())
    
    indices = hotspot_dfs_dict[hotspot_id]
    fixed_indices = fix_indices(indep.idx)
    for index in indices:
        hotspot[fixed_indices == index] = 1
    return hotspot.reshape(1, 1, -1, 1)

def default_hotspot_vector(L):
    return torch.zeros(L).reshape(1, 1, -1, 1)

def make_hotspot_id(item):
    chainID = item['CHAINID']
    pdb_ids = chainID.split(':')
    if len(pdb_ids) == 4:
        pdb_ids = pdb_ids[0:1] + pdb_ids[2:]
    hotspot_id = '-'.join(pdb_ids)
    return hotspot_id

def make_hotspot_id_distil(item):
    gene_id = item['gene_id']
    dna_seq = item['DNA sequence']
    hotspot_id = f'{gene_id}_{dna_seq}_00_init_min_bbcst_0001'
    return hotspot_id

def make_hotspot_vector_inference(indep, conf):
    # this is ugly but we gotta find the breaks
    first_ind_of_chain = [indep.idx[0]]
    for i, iplus1 in zip(indep.idx[:-1], indep.idx[1:]):
        if iplus1 != i + 1:
            first_ind_of_chain.append(iplus1)
    
    # parsing config
    indices_to_hotspot = []
    if 'hotspot_res_fixed' in conf:
        chains = conf.hotspot_res_fixed.split(';')
        for chain, iplus1 in zip(chains, first_ind_of_chain):
            residues = chain.split(':')[1]
            if len(residues) == 0:
                continue
            residues = residues.split(',')
            residues = [int(x) for x in residues]
            residues = [iplus1 + x - 1 for x in residues]
            indices_to_hotspot.extend(residues)
    
    elif 'hotspot_res_sample' in conf:
        chains = conf.hotspot_res_sample.split(';')
        for chain, iplus1 in zip(chains, first_ind_of_chain):
            residues = chain.split(':')[1]
            sample_size = int(conf.hotspot_res_sample_size)
            if len(residues) == 0:
                continue
            residues = residues.split(',')
            residues = [int(x) for x in residues]
            residues = [iplus1 + x - 1 for x in residues]
            residues = np.random.choice(residues, sample_size, replace=False)
            indices_to_hotspot.extend(residues)
            
    hotspot = torch.zeros(indep.length())
    for index in indices_to_hotspot:
        hotspot[indep.idx == index] = 1

    return hotspot.reshape(1, 1, -1, 1)