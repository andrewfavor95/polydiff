import sys
import numpy as np 
import random
from icecream import ic
from collections import OrderedDict
from pdb import set_trace
class ContigMap():
    '''
    New class for doing mapping.
    Supports multichain or multiple crops from a single receptor chain.
    Also supports indexing jump (+200) or not, based on contig input.
    Default chain outputs are inpainted chains as A (and B, C etc if multiple chains), and all fragments of receptor chain on the next one (generally B)
    Output chains can be specified. Sequence must be the same number of elements as in contig string
    '''
    def __init__(self, parsed_pdb, contigs=None, contig_atoms=None, inpaint_seq=None, inpaint_str=None, length=None, ref_idx=None, hal_idx=None, idx_rf=None, inpaint_seq_tensor=None, inpaint_str_tensor=None, topo=False, polymer_chains=None, use_old_receptor_system=True):
        #sanity checks
        if contigs is None and ref_idx is None:
            sys.exit("Must either specify a contig string or precise mapping")
        if idx_rf is not None or hal_idx is not None or ref_idx is not None:
            if idx_rf is None or hal_idx is None or ref_idx is None:
                sys.exit("If you're specifying specific contig mappings, the reference and output positions must be specified, AND the indexing for RoseTTAFold (idx_rf)")
        
        self.chain_order='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if length is not None:
            if '-' not in length:
                self.length = [int(length),int(length)+1]
            else:
                self.length = [int(length.split("-")[0]),int(length.split("-")[1])+1]
        else:
            self.length = None
        self.ref_idx = ref_idx
        self.hal_idx=hal_idx
        self.idx_rf=idx_rf
        self.inpaint_seq = ','.join(inpaint_seq).split(",") if inpaint_seq is not None else None
        self.inpaint_str = ','.join(inpaint_str).split(",") if inpaint_str is not None else None
        self.inpaint_seq_tensor=inpaint_seq_tensor
        self.inpaint_str_tensor=inpaint_str_tensor
        self.parsed_pdb = parsed_pdb
        self.topo=topo
        

        if ref_idx is None:
            #using default contig generation, which outputs in rosetta-like format
            self.contigs=contigs
            if contig_atoms is not None:
                self.contig_atoms={k:v.split(",") for k,v in eval(contig_atoms).items()}
            else:
                self.contig_atoms = None
            self.sampled_mask,self.contig_length,self.n_inpaint_chains = self.get_sampled_mask()
            
            self.receptor_chain = self.chain_order[self.n_inpaint_chains]
            
            self.receptor, self.receptor_hal, self.receptor_rf, self.inpaint, self.inpaint_hal, self.inpaint_rf, self.atomize_resnum2atomnames = self.expand_sampled_mask()
            if use_old_receptor_system:  # Using old weird way which is hard-coded for ppi... 
                # self.receptor, self.receptor_hal, self.receptor_rf, self.inpaint, self.inpaint_hal, self.inpaint_rf, self.atomize_resnum2atomnames = self.expand_sampled_mask()
                self.ref = self.inpaint + self.receptor
                self.hal = self.inpaint_hal + self.receptor_hal
                self.rf = self.inpaint_rf + self.receptor_rf 
            else:
                self.ref, self.hal, self.rf, self.atomize_resnum2atomnames = self.expand_sampled_mask_afav()

            self.atomize_indices2atomname = {self.ref.index(res_num):atom_names for res_num, atom_names in self.atomize_resnum2atomnames.items()}
            self.atomize_indices = list(self.atomize_indices2atomname.keys())

        else:
            #specifying precise mappings
            self.ref=ref_idx
            self.hal=hal_idx
            self.rf = rf_idx
        self.mask_1d = [False if i == ('_','_') else True for i in self.ref]

        #take care of sequence and structure masking
        if self.inpaint_seq_tensor is None:
            if self.inpaint_seq is not None:
                self.inpaint_seq = self.get_inpaint_seq_str(self.inpaint_seq)
            else:
                self.inpaint_seq = np.array([True if i != ('_','_') else False for i in self.ref])
        else:
            self.inpaint_seq = self.inpaint_seq_tensor

        if self.inpaint_str_tensor is None:
            if self.inpaint_str is not None:
                self.inpaint_str = self.get_inpaint_seq_str(self.inpaint_str)
            else:
                self.inpaint_str = np.array([True if i != ('_','_') else False for i in self.ref])
        else:
            self.inpaint_str = self.inpaint_str_tensor        


        #get 0-indexed input/output (for trb file)

        self.ref_idx0,self.hal_idx0, self.ref_idx0_inpaint, self.hal_idx0_inpaint, self.ref_idx0_receptor, self.hal_idx0_receptor=self.get_idx0(use_old_way=True)


        self.con_ref_pdb_idx=[i for i in self.ref if i != ('_','_')] 

        contig_list = self.contigs[0].strip().split()

        # if polymer_chains is None, then loop through and assign polymer_chains=['protein',...] for each chain
        if polymer_chains is None:
            print('WARNING: CURRENTLY ASSUMES EVERYTHING IS PROTEIN')
            print('TO DO: GO IN AND ASSIGN POLYMER TYPES BASED ON TEMPLATE/INPUT-PDB POLYMER TYPES')
            self.polymer_chains = ['protein' for chain_i in range(len(contig_list))]
            # set_trace()


        # else check that the input spec polymer_chains is same length as number of chains in design.
        else:
            assert len(polymer_chains)==len(contig_list) , "specified polymer chains need to match number of chains in design"
            self.polymer_chains = polymer_chains

        # apply inpaint_seq to stuff:
        # set_trace()



    def get_sampled_mask(self):
        '''
        Function to get a sampled mask from a contig.
        '''
        length_compatible=False
        count = 0
        while length_compatible is False:
            inpaint_chains=0
            contig_list = self.contigs[0].strip().split()
            # assert len(contig_list) == 1, f'contig_list with >1 element not curently supported: {contig_list}'
            sampled_mask = []
            sampled_mask_length = 0
            #allow receptor chain to be last in contig string
            if all([i[0].isalpha() for i in contig_list[-1].split(",")]):
                contig_list[-1] = f'{contig_list[-1]},0'

            for con in contig_list:
                if (all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0') or self.topo is True:
                    #receptor chain
                    sampled_mask.append(con)
                else:
                    inpaint_chains += 1
                    #chain to be inpainted. These are the only chains that count towards the length of the contig
                    subcons = con.split(",")
                    subcon_out = []
                    for subcon in subcons:
                        if subcon[0].isalpha():
                            subcon_out.append(subcon)
                            if '-' in subcon:
                                sampled_mask_length += (int(subcon.split("-")[1])-int(subcon.split("-")[0][1:])+1)
                            else:
                                sampled_mask_length += 1

                        else:
                            if '-' in subcon:
                                length_inpaint=random.randint(int(subcon.split("-")[0]),int(subcon.split("-")[1]))
                                subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                                sampled_mask_length += length_inpaint
                            elif subcon == '0':
                                subcon_out.append('0')
                            else:
                                length_inpaint=int(subcon)
                                subcon_out.append(f'{length_inpaint}-{length_inpaint}')
                                sampled_mask_length += int(subcon)
                    sampled_mask.append(','.join(subcon_out))
            #check length is compatible 
            if self.length is not None:
                if sampled_mask_length >= self.length[0] and sampled_mask_length < self.length[1]:
                    length_compatible = True
            else:
                length_compatible = True
            count+=1
            if count == 100000: #contig string incompatible with this length
                sys.exit("Contig string incompatible with --length range")
        return sampled_mask, sampled_mask_length, inpaint_chains

    def expand_sampled_mask(self):
        # chain_order='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        receptor = []
        inpaint = []
        receptor_hal = []
        inpaint_hal = []
        receptor_idx = 1
        inpaint_idx = 1
        inpaint_chain_idx=-1
        receptor_chain_break=[]
        inpaint_chain_break = []
        atomize_resnum2atomnames = {}
        for con in self.sampled_mask:
            if (all([i[0].isalpha() for i in con.split(",")[:-1]]) and con.split(",")[-1] == '0') or self.topo is True:
                #receptor chain
                subcons = con.split(",")[:-1]
                assert all([i[0] == subcons[0][0] for i in subcons]), "If specifying fragmented receptor in a single block of the contig string, they MUST derive from the same chain"
                assert all(int(subcons[i].split("-")[0][1:]) < int(subcons[i+1].split("-")[0][1:]) for i in range(len(subcons)-1)), "If specifying multiple fragments from the same chain, pdb indices must be in ascending order!"
                for idx, subcon in enumerate(subcons):
                    ref_to_add = [(subcon[0], i) for i in np.arange(int(subcon.split("-")[0][1:]),int(subcon.split("-")[1])+1)]
                    receptor.extend(ref_to_add)
                    receptor_hal.extend([(self.receptor_chain,i) for i in np.arange(receptor_idx, receptor_idx+len(ref_to_add))])
                    receptor_idx += len(ref_to_add)
                    if self.contig_atoms is not None:
                        atomize_resnum2atomnames.update({(k[0], int(k[1:])):v for k, v in self.contig_atoms.items() if (k[0], int(k[1:])) in receptor})
                    if idx != len(subcons)-1:
                        idx_jump = int(subcons[idx+1].split("-")[0][1:]) - int(subcon.split("-")[1]) -1 
                        receptor_chain_break.append((receptor_idx-1,idx_jump)) #actual chain break in pdb chain
                    else:
                        receptor_chain_break.append((receptor_idx-1,200)) #200 aa chain break 
            else:
                inpaint_chain_idx += 1
                for subcon in con.split(","):
                    if subcon[0].isalpha(): # this is a part of the motif because the first element of the contig is the chain letter
                        ref_to_add=[(subcon[0], i) for i in np.arange(int(subcon.split("-")[0][1:]),int(subcon.split("-")[1])+1)]
                        inpaint.extend(ref_to_add)
                        inpaint_hal.extend([(self.chain_order[inpaint_chain_idx], i) for i in np.arange(inpaint_idx,inpaint_idx+len(ref_to_add))])
                        inpaint_idx += len(ref_to_add)
                        if self.contig_atoms is not None:
                            atomize_resnum2atomnames.update({(k[0], int(k[1:])):v for k, v in self.contig_atoms.items() if (k[0], int(k[1:])) in inpaint})
                    else:
                        inpaint.extend([('_','_')] * int(subcon.split("-")[0]))
                        inpaint_hal.extend([(self.chain_order[inpaint_chain_idx], i) for i in np.arange(inpaint_idx,inpaint_idx+int(subcon.split("-")[0]))])
                        inpaint_idx += int(subcon.split("-")[0])
                inpaint_chain_break.append((inpaint_idx-1,200))

        
        if self.topo is True or inpaint_hal == []:
            receptor_hal = [(i[0], i[1]) for i in receptor_hal]
        else:        
            receptor_hal = [(i[0], i[1] + inpaint_hal[-1][1]) for i in receptor_hal] #rosetta-like numbering
        #get rf indexes, with chain breaks
        inpaint_rf = np.arange(0,len(inpaint))
        receptor_rf = np.arange(len(inpaint)+200,len(inpaint)+len(receptor)+200)
        for ch_break in inpaint_chain_break[:-1]:
            receptor_rf[:] += 200
            inpaint_rf[ch_break[0]:] += ch_break[1]
        for ch_break in receptor_chain_break[:-1]:
            receptor_rf[ch_break[0]:] += ch_break[1]

        # set_trace()

        return receptor, receptor_hal, receptor_rf.tolist(), inpaint, inpaint_hal, inpaint_rf.tolist(), atomize_resnum2atomnames


    def expand_sampled_mask_afav(self):

        """
        outputs:


        self.ref, self.hal, self.rf, self.atomize_resnum2atomnames

        full_ref, full_hal, full_rf, self.atomize_resnum2atomnames
            full_ref = inpaint + receptor
            full_hal = inpaint_hal + receptor_hal
            full_rf = inpaint_rf + receptor_rf 

        """


        chain_order='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # receptor = []
        # inpaint = []
        # receptor_hal = []
        # inpaint_hal = []
        # receptor_idx = 1
        # inpaint_idx = 1
        # inpaint_chain_idx=-1
        # receptor_chain_break=[]
        # inpaint_chain_break = []
        # atomize_resnum2atomnames = {}

        # receptor = []
        # inpaint = []
        full_ref = []
        # receptor_hal = []
        # inpaint_hal = []
        full_hal = []
        # receptor_idx = 1
        # inpaint_idx = 1
        full_idx = 1
        # inpaint_chain_idx=-1
        full_chain_idx=-1
        # receptor_chain_break=[]
        # inpaint_chain_break = []
        full_chain_break=[]
        atomize_resnum2atomnames = {}
        for con in self.sampled_mask:

            full_chain_idx += 1
            for subcon in con.split(","):
                if subcon[0].isalpha(): # this is a part of the motif because the first element of the contig is the chain letter
                    ref_to_add=[(subcon[0], i) for i in np.arange(int(subcon.split("-")[0][1:]),int(subcon.split("-")[1])+1)]
                    full_ref.extend(ref_to_add)
                    
                    full_hal.extend([(self.chain_order[full_chain_idx], i) for i in np.arange(full_idx,full_idx+len(ref_to_add))])
                    full_idx += len(ref_to_add)
                    if self.contig_atoms is not None:
                        atomize_resnum2atomnames.update({(k[0], int(k[1:])):v for k, v in self.contig_atoms.items() if (k[0], int(k[1:])) in full_ref})
                else:
                    full_ref.extend([('_','_')] * int(subcon.split("-")[0]))

                    full_hal.extend([(self.chain_order[full_chain_idx], i) for i in np.arange(full_idx,full_idx+int(subcon.split("-")[0]))])

                    full_idx += int(subcon.split("-")[0])
            full_chain_break.append((full_idx-1,200))

        full_rf = np.arange(0,len(full_ref))
        for ch_break in full_chain_break[:-1]:
            full_rf[ch_break[0]:] += ch_break[1]

        return full_ref, full_hal, full_rf.tolist(), atomize_resnum2atomnames


    def get_inpaint_seq_str(self, inpaint_s):
        '''
        function to generate inpaint_str or inpaint_seq masks specific to this contig
        '''
        s_mask = np.copy(self.mask_1d)
        inpaint_s_list = []
        for i in inpaint_s:
            if '-' in i:
                inpaint_s_list.extend([(i[0],p) for p in range(int(i.split("-")[0][1:]), int(i.split("-")[1])+1)])
            else:
                inpaint_s_list.append((i[0],int(i[1:])))

        for res in inpaint_s_list:
            if res in self.ref:
                s_mask[self.ref.index(res)] = False #mask this residue
        
        return np.array(s_mask) 

    def get_idx0(self, use_old_way=True):

        ref_idx0=[]
        hal_idx0=[]
        for idx, val in enumerate(self.ref):
            if val != ('_','_'):
                assert val in self.parsed_pdb['pdb_idx'],f"{val} is not in pdb file!"
                hal_idx0.append(idx)
                ref_idx0.append(self.parsed_pdb['pdb_idx'].index(val))

        ref_idx0_inpaint=[]
        hal_idx0_inpaint=[]
        for idx, val in enumerate(self.inpaint):
            if val != ('_','_'):
                hal_idx0_inpaint.append(idx)
                ref_idx0_inpaint.append(self.parsed_pdb['pdb_idx'].index(val))

        ref_idx0_receptor=[]
        hal_idx0_receptor=[]
        for idx, val in enumerate(self.receptor):
            if val != ('_','_'):
                hal_idx0_receptor.append(idx)
                ref_idx0_receptor.append(self.parsed_pdb['pdb_idx'].index(val))

        return ref_idx0, hal_idx0, ref_idx0_inpaint, hal_idx0_inpaint, ref_idx0_receptor, hal_idx0_receptor

    def get_mappings(self):

        mappings = {}
        mappings['con_ref_pdb_idx'] = [i for i in self.ref if i != ('_','_')]
        mappings['con_hal_pdb_idx'] = [self.hal[i] for i in range(len(self.hal)) if self.ref[i] != ("_","_")]
        mappings['con_ref_idx0'] = np.array(self.ref_idx0)
        mappings['con_hal_idx0'] = np.array(self.hal_idx0)

        if self.inpaint != self.ref:
            mappings['complex_con_ref_pdb_idx'] = [i for i in self.ref if i != ("_","_")]
            mappings['complex_con_hal_pdb_idx'] = [self.hal[i] for i in range(len(self.hal)) if self.ref[i] != ("_","_")]
            mappings['receptor_con_ref_pdb_idx'] = [i for i in self.receptor if i != ("_","_")]
            mappings['receptor_con_hal_pdb_idx'] = [self.receptor_hal[i] for i in range(len(self.receptor_hal)) if self.receptor[i] != ("_","_")]
            mappings['complex_con_ref_idx0'] = np.array(self.ref_idx0)
            mappings['complex_con_hal_idx0'] = np.array(self.hal_idx0)
            mappings['receptor_con_ref_idx0'] = np.array(self.ref_idx0_receptor)
            mappings['receptor_con_hal_idx0'] = np.array(self.hal_idx0_receptor)

        mappings['inpaint_str'] = self.inpaint_str
        mappings['inpaint_seq'] = self.inpaint_seq
        mappings['sampled_mask'] = self.sampled_mask
        mappings['mask_1d'] = self.mask_1d
        mappings['atomize_indices2atomname'] = self.atomize_indices2atomname

        return mappings



def select_and_apply_contig(indep, contig_conf, idx_pdb):
    if contig_conf.contig_type == 'DNA_Duplex_Protein_Monomer':
        return DNA_Duplex_Protein_Monomer(indep, contig_conf, idx_pdb)
    elif contig_conf.contig_type == 'Partial_Diffusion_DNA_Duplex_Protein_Monomer':
        return Partial_Diffusion_DNA_Duplex_Protein_Monomer(indep, contig_conf, idx_pdb)
    elif contig_conf.contig_type == 'Single_Protein_Chain_Arbitrary_Context':
        return Single_Protein_Chain_Arbitrary_Context(indep, contig_conf, idx_pdb)


def process_insert(group, indep):
    if '-' in group:
        split = group.split('-')
        insert_min, insert_max = int(split[0]), int(split[1])
        insert = random.randint(insert_min, insert_max)
    else:
        insert = int(group)
    insert_seq = torch.full((insert,), MASKINDEX)
    insert_xyz = INIT_CRDS.clone().to(indep.xyz.device).unsqueeze(0)
    insert_xyz = insert_xyz.repeat(insert, 1, 1)

    return insert_seq, insert_xyz


def process_copy(group, indep, pdbidx_index_map):
    if '-' in group:
        split = group.split('-')
        copy_start, copy_end = int(split[0]), int(split[1])
    else:
        copy_start, copy_end = int(group), int(group)
    
    try:
        original_index_start, original_index_end = pdbidx_index_map[copy_start], pdbidx_index_map[copy_end]
    except:
        original_index_start = pdbidx_index_map[copy_start]
        original_index_end = original_index_start

    return indep.seq[original_index_start:original_index_end + 1], indep.xyz[original_index_start:original_index_end + 1]
    

    
def DNA_Duplex_Protein_Monomer(indep, contig_conf, idx_pdb):

    chainid_list = 'ABCDEFG123456789'
    
    # protein_chains = ['A']
    # dna_chains = ['B','C','D','E']
    split_contig = contig_conf['contigs'][0].split(' ')
    assert len(split_contig) == len(contig_conf['polymer_chains'])
    protein_chains = []
    dna_chains = []
    rna_chains = []

    protein_contig = []
    dna_contig = []
    rna_contig = []
    for i, (polymer_type, subcontig) in enumerate(zip(contig_conf['polymer_chains'],split_contig)):

        if polymer_type in ['protein','prot','aa','AA','a','A']:
            protein_chains.append(chainid_list[i])
            protein_contig.append(subcontig)

        elif polymer_type in ['dna','DNA','d','D']:
            dna_chains.append(chainid_list[i])
            dna_contig.append(subcontig)

        elif polymer_type in ['rna','RNA','r','R']:
            rna_chains.append(chainid_list[i])
            rna_contig.append(subcontig)

    protein_contig = ' '.join(protein_contig)
    dna_contig = ' '.join(dna_contig)
    rna_contig = ' '.join(rna_contig)

    assert len(protein_chains) == 1, "this code currently assumes protein monomers"

    chain_pdbidx_index_map = {}
    chain_index_map = {}

    for chainid in dna_chains + protein_chains:
        chain_pdbidx_index_map[chainid] = {}
        chain_index_map[chainid] = []

    for i, (chainid, pdbidx) in enumerate(idx_pdb):
        chain_pdbidx_index_map[chainid][pdbidx] = i
        chain_index_map[chainid].append(i)

    chain_index_map = {key: np.array(value) for key, value in chain_index_map.items()}

    seq = []
    xyz = []
    idx = []
    is_diffused = []
    
    last_start = 0
    chain_lengths = []
    # ipdb.set_trace()
    
    for chainid in chainid_list:
        if chainid in dna_chains:
            chain_indices, L = chain_index_map[chainid], len(chain_index_map[chainid])
            seq.append(indep.seq[chain_indices])
            xyz.append(indep.xyz[chain_indices])
            idx.append(torch.arange(last_start, last_start + L))
            is_diffused.extend([False] * L)
            last_start += 200 + L
            chain_lengths.append(L)

        elif chainid in protein_chains:
            L = 0

            for i, group in enumerate(protein_contig.split(',')):
                if group[0] == 'i': # insert
                    # Logic for forcing total length if you're adding the final insert
                    # Currently only works if the final contig in the contig string is an insertion, and does not check
                    # for if the added length violates the actual desired length at that position. 
                    if i == len(protein_contig.split(',')) - 1 and 'total_protein_length' in contig_conf:
                        insert_seq, insert_xyz = process_insert(str(int(total_protein_length) - L), indep)
                        is_diffused.extend([True] * len(insert_seq))
                    else:
                        insert_seq, insert_xyz = process_insert(group[1:], indep)
                        is_diffused.extend([True] * len(insert_seq))
                else:
                    insert_seq, insert_xyz = process_copy(group, indep, chain_pdbidx_index_map[chainid])
                    is_diffused.extend([False] * len(insert_seq))

                seq.append(insert_seq)
                xyz.append(insert_xyz)
                idx.append(torch.arange(last_start, last_start + len(insert_seq)))
                last_start += len(insert_seq)
                L += len(insert_seq)

            last_start += 200
            chain_lengths.append(L)

    ######## This code block only works because there is just one protein chain
    protein_start = 0
    i = 0
    for chain in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789':
        if chain in protein_chains:
            protein_end = protein_start + chain_lengths[i]
            break
        elif chain in dna_chains:
            protein_start += chain_lengths[i]
            i += 1
    # for i, chainid in enumerate('ABCDEFG'):
    #     if chainid in protein_chains:
    #         protein_end = protein_start + chain_lengths[i]
    #     elif chainid in dna_chains:
    #         protein_start += chain_lengths[i]
    ########

    total_length = sum(chain_lengths)
    bond_feats = torch.full((total_length, total_length), 0).long()
    bond_feats[protein_start:protein_end, protein_start:protein_end] = get_protein_bond_feats(protein_end-protein_start)

    same_chain = same_chain_2d_from_Ls(chain_lengths)
    
    # same_chain = torch.full((total_length, total_length), 0).long()
    # for start, end in zip([0] + chain_lengths[:-1], chain_lengths):
    #     same_chain[start:end, start:end] = 1
    
    terminus_type = torch.zeros(total_length)
    terminus_type[protein_start] = N_TERMINUS
    terminus_type[protein_end-1] = C_TERMINUS

    is_sm = torch.zeros(total_length).bool()

    seq = torch.cat(seq)
    xyz = torch.cat(xyz)
    idx = torch.cat(idx)
    is_diffused = torch.Tensor(is_diffused).bool()

    # ipdb.set_trace()
    return Indep(seq,
                 xyz,
                 idx,
                 bond_feats,
                 indep.chirals,
                 indep.atom_frames,
                 same_chain,
                 is_sm,
                 terminus_type), is_diffused
    
# def DNA_Duplex_Protein_Monomer(indep, contig_conf, idx_pdb):

#     ipdb.set_trace()
#     contig_conf.protein_chains = ['A']
#     contig_conf.dna_chains = ['B','C','D','E']

#     assert len(contig_conf.protein_chains) == 1, "this code currently assumes protein monomers"

#     chain_pdbidx_index_map = {}
#     chain_index_map = {}

#     for chainid in contig_conf.dna_chains + contig_conf.protein_chains:
#         chain_pdbidx_index_map[chainid] = {}
#         chain_index_map[chainid] = []

#     for i, (chainid, pdbidx) in enumerate(idx_pdb):
#         chain_pdbidx_index_map[chainid][pdbidx] = i
#         chain_index_map[chainid].append(i)

#     chain_index_map = {key: np.array(value) for key, value in chain_index_map.items()}

#     seq = []
#     xyz = []
#     idx = []
#     is_diffused = []
    
#     last_start = 0
#     chain_lengths = []
    
#     for chainid in 'ABCDEFG123456789':
#         if chainid in contig_conf.dna_chains:
#             chain_indices, L = chain_index_map[chainid], len(chain_index_map[chainid])
#             seq.append(indep.seq[chain_indices])
#             xyz.append(indep.xyz[chain_indices])
#             idx.append(torch.arange(last_start, last_start + L))
#             is_diffused.extend([False] * L)
#             last_start += 200 + L
#             chain_lengths.append(L)

#         elif chainid in contig_conf.protein_chains:
#             L = 0

#             for i, group in enumerate(contig_conf.protein_contig.split(',')):
#                 if group[0] == 'i': # insert
#                     # Logic for forcing total length if you're adding the final insert
#                     # Currently only works if the final contig in the contig string is an insertion, and does not check
#                     # for if the added length violates the actual desired length at that position. 
#                     if i == len(contig_conf.protein_contig.split(',')) - 1 and 'total_protein_length' in contig_conf:
#                         insert_seq, insert_xyz = process_insert(str(int(contig_conf.total_protein_length) - L), indep)
#                         is_diffused.extend([True] * len(insert_seq))
#                     else:
#                         insert_seq, insert_xyz = process_insert(group[1:], indep)
#                         is_diffused.extend([True] * len(insert_seq))
#                 else:
#                     insert_seq, insert_xyz = process_copy(group, indep, chain_pdbidx_index_map[chainid])
#                     is_diffused.extend([False] * len(insert_seq))

#                 seq.append(insert_seq)
#                 xyz.append(insert_xyz)
#                 idx.append(torch.arange(last_start, last_start + len(insert_seq)))
#                 last_start += len(insert_seq)
#                 L += len(insert_seq)

#             last_start += 200
#             chain_lengths.append(L)

#     ######## This code block only works because there is just one protein chain
#     protein_start = 0
#     i = 0
#     for chain in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789':
#         if chain in contig_conf.protein_chains:
#             protein_end = protein_start + chain_lengths[i]
#             break
#         elif chain in contig_conf.dna_chains:
#             protein_start += chain_lengths[i]
#             i += 1
#     # for i, chainid in enumerate('ABCDEFG'):
#     #     if chainid in contig_conf.protein_chains:
#     #         protein_end = protein_start + chain_lengths[i]
#     #     elif chainid in contig_conf.dna_chains:
#     #         protein_start += chain_lengths[i]
#     ########

#     total_length = sum(chain_lengths)
#     bond_feats = torch.full((total_length, total_length), 0).long()
#     bond_feats[protein_start:protein_end, protein_start:protein_end] = get_protein_bond_feats(protein_end-protein_start)

#     same_chain = same_chain_2d_from_Ls(chain_lengths)
    
#     # same_chain = torch.full((total_length, total_length), 0).long()
#     # for start, end in zip([0] + chain_lengths[:-1], chain_lengths):
#     #     same_chain[start:end, start:end] = 1
    
#     terminus_type = torch.zeros(total_length)
#     terminus_type[protein_start] = N_TERMINUS
#     terminus_type[protein_end-1] = C_TERMINUS

#     is_sm = torch.zeros(total_length).bool()

#     seq = torch.cat(seq)
#     xyz = torch.cat(xyz)
#     idx = torch.cat(idx)
#     is_diffused = torch.Tensor(is_diffused).bool()

#     ipdb.set_trace()
#     return Indep(seq,
#                  xyz,
#                  idx,
#                  bond_feats,
#                  indep.chirals,
#                  indep.atom_frames,
#                  same_chain,
#                  is_sm,
#                  terminus_type), is_diffused

    # handle bond feats 

def Single_Protein_Chain_Arbitrary_Context(indep, contig_conf, idx_pdb):
    """
    Assumes all your chains are either only context chains (the entire chain should be copied over as context, as a separate chain, or is the single diffusion chain (all parts to be concatenated to a single protein chain))
    """
    chain_pdbidx_index_map = {}
    chain_index_map = {}

    for chainid in contig_conf.context_chains + contig_conf.diffusion_chain:
        chain_pdbidx_index_map[chainid] = {}
        chain_index_map[chainid] = []

    for i, (chainid, pdbidx) in enumerate(idx_pdb):
        chain_pdbidx_index_map[chainid][pdbidx] = i
        chain_index_map[chainid].append(i)

    chain_index_map = {key: np.array(value) for key, value in chain_index_map.items()}

    seq = []
    xyz = []
    idx = []
    is_diffused = []
    
    last_start = 0
    chain_lengths = []

    for chainid in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789':
        if chainid in contig_conf.context_chains:
            chain_indices, L = chain_index_map[chainid], len(chain_index_map[chainid])
            seq.append(indep.seq[chain_indices])
            xyz.append(indep.xyz[chain_indices])
            idx.append(torch.arange(last_start, last_start + L))
            is_diffused.extend([False] * L)
            last_start += 200 + L
            chain_lengths.append(L)

        elif chainid in contig_conf.diffusion_chain:
            L = 0
            for i, group in enumerate(contig_conf.protein_contig.split(',')):
                if group[0] == 'i': # insert
                    insert_seq, insert_xyz = process_insert(group[1:], indep)
                    is_diffused.extend([True] * len(insert_seq))
                else:
                    insert_seq, insert_xyz = process_copy(group, indep, chain_pdbidx_index_map[chainid])
                    is_diffused.extend([False] * len(insert_seq))

                seq.append(insert_seq)
                xyz.append(insert_xyz)
                idx.append(torch.arange(last_start, last_start + len(insert_seq)))
                last_start += len(insert_seq)
                L += len(insert_seq)

            last_start += 200
            chain_lengths.append(L)
    
    seq = torch.cat(seq)
    xyz = torch.cat(xyz)
    idx = torch.cat(idx)
    is_diffused = torch.Tensor(is_diffused).bool()

    curr_ind = 0
    total_length = sum(chain_lengths)
    bond_feats = torch.full((total_length, total_length), 0).long()
    terminus_type = torch.zeros(total_length)
    chainids = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
    for i, L in enumerate(chain_lengths):
        if chainids[i] in contig_conf.diffusion_chain:
            bond_feats[curr_ind : curr_ind + L, curr_ind : curr_ind + L] = get_protein_bond_feats(L)
            terminus_type[curr_ind] = N_TERMINUS
            terminus_type[curr_ind + L] = C_TERMINUS

        curr_ind += L

    is_sm = torch.zeros(total_length).bool()
    same_chain = same_chain_2d_from_Ls(chain_lengths)


    return Indep(seq,
                 xyz,
                 idx,
                 bond_feats,
                 indep.chirals,
                 indep.atom_frames,
                 same_chain,
                 is_sm,
                 terminus_type), is_diffused

def Partial_Diffusion_DNA_Duplex_Protein_Monomer(indep, contig_conf, idx_pdb):
    assert len(contig_conf.protein_chains) == 1, "this code currently assumes protein monomers"
    chain_pdbidx_index_map = {}
    chain_index_map = {}

    for chainid in contig_conf.dna_chains + contig_conf.protein_chains:
        chain_pdbidx_index_map[chainid] = {}
        chain_index_map[chainid] = []

    for i, (chainid, pdbidx) in enumerate(idx_pdb):
        chain_pdbidx_index_map[chainid][pdbidx] = i
        chain_index_map[chainid].append(i)

    chain_index_map = {key: np.array(value) for key, value in chain_index_map.items()}

    seq = []
    xyz = []
    idx = []
    is_diffused = []
    
    last_start = 0
    chain_lengths = []

    for chainid in 'ABCDEFG':
        if chainid in contig_conf.dna_chains:
            chain_indices, L = chain_index_map[chainid], len(chain_index_map[chainid])
            seq.append(indep.seq[chain_indices])
            xyz.append(indep.xyz[chain_indices])
            idx.append(torch.arange(last_start, last_start + L))
            is_diffused.extend([False] * L)
            last_start += 200 + L
            chain_lengths.append(L)

        elif chainid in contig_conf.protein_chains:
            prot_pdbidx_start = chain_index_map[chainid][0]
            chain_indices, L = chain_index_map[chainid], len(chain_index_map[chainid])

            prot_is_diffused = [True] * L
            seq.append(torch.full((L,), MASKINDEX))
            xyz_to_append = torch.full((L, NTOTAL, 3), np.nan)
            xyz_to_append[:, :3] = indep.xyz[chain_indices, :3]
            xyz.append(xyz_to_append) # only take the N, C, Ca 
            idx.append(torch.arange(last_start, last_start + L))

            if 'motif_residues' in contig_conf:
                for motif_residue_index in contig_conf.motif_residues.split(','):
                    prot_is_diffused[int(motif_residue_index) - prot_pdbidx_start] = False
            
            is_diffused.extend(prot_is_diffused)
            last_start += 200 + L
            chain_lengths.append(L)

    ######## This code block only works because there is just one protein chain
    protein_start = 0
    i = 0
    for chain in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if chain in contig_conf.protein_chains:
            protein_end = protein_start + chain_lengths[i]
            break
        elif chain in contig_conf.dna_chains:
            protein_start += chain_lengths[i]
            i += 1
    # for i, chainid in enumerate('ABCDEFG'):
    #     if chainid in contig_conf.protein_chains:
    #         protein_end = protein_start + chain_lengths[i]
    #     elif chainid in contig_conf.dna_chains:
    #         protein_start += chain_lengths[i]
    ########

    total_length = sum(chain_lengths)
    bond_feats = torch.full((total_length, total_length), 0).long()
    bond_feats[protein_start:protein_end, protein_start:protein_end] = get_protein_bond_feats(protein_end-protein_start)

    same_chain = same_chain_2d_from_Ls(chain_lengths)
    
    # same_chain = torch.full((total_length, total_length), 0).long()
    # for start, end in zip([0] + chain_lengths[:-1], chain_lengths):
    #     same_chain[start:end, start:end] = 1
    
    terminus_type = torch.zeros(total_length)
    terminus_type[protein_start] = N_TERMINUS
    terminus_type[protein_end-1] = C_TERMINUS

    is_sm = torch.zeros(total_length).bool()

    seq = torch.cat(seq)
    xyz = torch.cat(xyz)
    idx = torch.cat(idx)
    is_diffused = torch.Tensor(is_diffused).bool()

    return Indep(seq,
                 xyz,
                 idx,
                 bond_feats,
                 indep.chirals,
                 indep.atom_frames,
                 same_chain,
                 is_sm,
                 terminus_type), is_diffused


