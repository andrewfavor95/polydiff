import torch
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
# import time

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




def make_poly_hotspots_vec(indep, 
                            contact_cutoff = 8.0,
                            cum_contact_lim = 16.0,
                            max_hotspots = 3,
                            min_close_contacts = 3,
                            seq_dist_cutoff = 3,
                            p_interchain=0.5,
                            cum_contact_dt=1.0,
                            png_filepath=None):
                            # cum_contact_dt=0.2,

    """
    Average execution time for everything

    """

    contact_type_to_int = {
                    'protein_protein': 0, 
                    'dna_dna': 1, 
                    'rna_rna': 2, 
                    'protein_dna': 3, 
                    'dna_protein': 3, 
                    'protein_rna': 4, 
                    'rna_protein': 4, 
                    'dna_rna': 5, 
                    'rna_dna': 5, 
                    }

    # start_time = time.time()

 
    N = indep.seq.size(0)
    multi_chain = (indep.same_chain[0,:].sum() < N)

    valid_atoms = indep.xyz[:,:4,:]
    centroid = torch.nanmean(valid_atoms, dim=1)

    n_hotspots_dict = {contact_type: 0 for contact_type in contact_type_to_int.keys()}

    new_seq_dist_cutoff = seq_dist_cutoff
    kernel_size = 3

    smoothing_kernel = torch.full((1, 1, kernel_size, kernel_size), 1/(kernel_size*kernel_size)).float()
    kernel_padding=int((kernel_size-1)/2)
    centroid_contact_dist = torch.cdist(centroid,centroid)
    cum_contact_matrix = torch.zeros_like(centroid_contact_dist)
    
    for n in range(int(cum_contact_lim/cum_contact_dt)):
        cum_contact_matrix += cum_contact_dt*(centroid_contact_dist <= (n+1)*cum_contact_dt )

    # First filter: super close neighbors
    close_contact_matrix = (centroid_contact_dist <= contact_cutoff)
    
    # Second filter: remove close neuighbors in sequence space
    seq_neighbors = torch.le(torch.abs(torch.arange(N)[:,None]-torch.arange(N)[None,:]), new_seq_dist_cutoff)
    close_contact_matrix = torch.logical_and(close_contact_matrix , ~seq_neighbors)
    cum_contact_matrix = torch.logical_and(cum_contact_matrix, ~seq_neighbors)


    cum_contact_smooth = torch.nn.functional.conv2d(
            cum_contact_matrix.unsqueeze(0).unsqueeze(0).float(), 
            smoothing_kernel, 
            padding=kernel_padding
        ).squeeze(0).squeeze(0)

    cum_contact_matrix = ~seq_neighbors * cum_contact_smooth


    if multi_chain and (np.random.rand() <= p_interchain):
        # close_contact_matrix = torch.logical_and(close_contact_matrix , torch.logical_not(indep.same_chain))
        cum_contact_matrix =  cum_contact_matrix * torch.logical_not(indep.same_chain)

    num_close_contacts = close_contact_matrix.sum(dim=1)
    poly_hotspot_template = torch.zeros((N,6)).long()

    all_hotspot_inds = []
    all_hotspot_ncontacts = []
    
    all_contact_types = []
    contact_type_list = []

    weight_mat = cum_contact_matrix

    flat_matrix = weight_mat.view(-1)

    if flat_matrix.sum()<0.01:
        return poly_hotspot_template

    sampled_indices = torch.multinomial(flat_matrix, max_hotspots*4, replacement=False)

    rows = sampled_indices // N
    cols = sampled_indices % N

    for resi_i, resi_j in zip(rows,cols):

        if indep.is_protein[resi_i]:
            poly_i = 'protein'
        elif indep.is_dna[resi_i]:
            poly_i = 'dna'
        elif indep.is_rna[resi_i]:
            poly_i = 'rna'
        else:
            continue

        if indep.is_protein[resi_j]:
            poly_j = 'protein'
        elif indep.is_dna[resi_j]:
            poly_j = 'dna'
        elif indep.is_rna[resi_j]:
            poly_j = 'rna'
        else:
            continue

        all_hotspot_inds.append(resi_i)
        all_hotspot_inds.append(resi_j)

        all_hotspot_ncontacts.append(num_close_contacts[resi_i])
        all_hotspot_ncontacts.append(num_close_contacts[resi_j])

        all_contact_types.append(f'{poly_i}_{poly_j}')
        all_contact_types.append(f'{poly_j}_{poly_i}')


    # Pairing the elements and sorting
    paired_sorted = sorted(zip(all_hotspot_ncontacts, all_hotspot_inds, all_contact_types), key=lambda x: x[0], reverse=True)

    # Unzipping the pairs
    all_hotspot_ncontacts_sorted, all_hotspot_inds_sorted, all_contact_types_sorted = zip(*paired_sorted)

    # Convert the tuples back to lists (optional)
    all_hotspot_ncontacts_sorted = list(all_hotspot_ncontacts_sorted)
    all_hotspot_inds_sorted = list(all_hotspot_inds_sorted)
    all_contact_types_sorted = list(all_contact_types_sorted)


    all_chosen_hotspot_inds = []
    for i, (ncontacts_i, resi_i, contact_type_i) in enumerate(zip(all_hotspot_ncontacts_sorted, all_hotspot_inds_sorted, all_contact_types_sorted)):
        poly_i, poly_j = contact_type_i.split('_')
        if (n_hotspots_dict[f'{poly_i}_{poly_j}']  < max_hotspots) and (ncontacts_i >= min_close_contacts):
            all_chosen_hotspot_inds.append(resi_i)
            poly_hotspot_template[resi_i, contact_type_to_int[contact_type_i]] = 1
            n_hotspots_dict[contact_type_i] += 1




    # End timing
    # end_time = time.time()

    # Calculate and print the execution time
    # execution_time = end_time - start_time
    # print(f"EXECUTION TIME TO MAKE POLY HOTSPOTS: {execution_time} seconds")

    # If we want to save png outputs for debugging
    if png_filepath:

        print(png_filepath)
        
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        # First subplot
        cax1 = axs[0].imshow(centroid_contact_dist, cmap='cool')
        axs[0].set_title('Centroid Contact Distance')

        cax2 = axs[1].imshow(cum_contact_matrix, cmap='cool')
        axs[1].set_title('Smoothed smoothed cum contact')
        for r,c in zip(rows,cols):
            # print(f'color magenta, sele and resi {r+1} ; ')
            # print(f'color orange, sele and resi {c+1} ; ')
            axs[1].axvline(x=r, color='red', linewidth=1)
            axs[1].axhline(y=c, color='orangered', linewidth=1)

        # Second subplot
        cax3 = axs[2].imshow(close_contact_matrix, cmap='cool')
        axs[2].set_title('Close Contact Matrix')

        # Third subplot
        cax4 = axs[3].imshow(centroid_contact_dist, cmap='cool')
        axs[3].set_title('Centroid Contact with Hotspots')

        # Overlay red lines on the third subplot
        for i in all_hotspot_inds:
            # if is_valid:
            print(f'color white, sele and resi {i+1} ; ')
            axs[3].axhline(y=i, color='red', linewidth=1)
            axs[3].axvline(x=i, color='red', linewidth=1)

        # Fifth subplot
        cax5 = axs[4].imshow(centroid_contact_dist, cmap='cool')
        axs[4].set_title('Chosen hotspots')



        for i in all_chosen_hotspot_inds:
            print(f'color magenta, sele and resi {i+1} ; ')
            axs[4].axhline(y=i, color='red', linewidth=1)
            axs[4].axvline(x=i, color='red', linewidth=1)

        # Add a colorbar for the whole figure
        plt.tight_layout()
        fig.colorbar(cax5, ax=axs.ravel().tolist())

        plt.savefig(png_filepath)
        plt.close(fig)

    # set_trace()


        

    return poly_hotspot_template





