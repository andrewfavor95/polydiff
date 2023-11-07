import sys
import numpy as np
import scipy
import scipy.spatial
import string
import os,re
import random
import util
import gzip
import rf2aa.util
import rf2aa.parsers

to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    #print(filename)
    
    if filename.split('.')[-1] == 'gz':
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, 'r')

    # read file line by line
    for line in fp:

        # skip labels
        if line[0] == '>':
            continue
            
        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins


# read and extract xyz coords of N,Ca,C atoms
# from a PDB file

def parse_pdb(filename, xyz27=False,seq=False):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, xyz27, seq)

#'''
def parse_pdb_lines(lines, xyz27, seq, get_aa=util.aa2num.get):

    # indices of residues observed in the structure
    idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    # 4 BB + up to 10 SC atoms
    if xyz27:
        xyz = np.full((len(idx_s), 27, 3), np.nan, dtype=np.float32)
    else:
        xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(util.aa2long[get_aa(aa)]):
            if tgtatm and tgtatm.strip() == atom.strip():
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0 
    if not seq:
        return xyz,mask,np.array(idx_s)
    else:
        return xyz,mask,np.array(idx_s),np.array(seq)

#'''

'''
def parse_pdb_lines(lines):

    # indices of residues observed in the structure
    #idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    idx_s = [int(r[0]) for r in res]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz,mask,np.array(idx_s), np.array(seq)
'''


def parse_templates(item, params):

    # init FFindexDB of templates
    ### and extract template IDs
    ### present in the DB
    ffdb = FFindexDB(read_index(params['FFDB']+'_pdb.ffindex'),
                     read_data(params['FFDB']+'_pdb.ffdata'))
    #ffids = set([i.name for i in ffdb.index])

    # process tabulated hhsearch output to get
    # matched positions and positional scores
    infile = params['DIR']+'/hhr/'+item[-2:]+'/'+item+'.atab'
    hits = []
    for l in open(infile, "r").readlines():
        if l[0]=='>':
            key = l[1:].split()[0]
            hits.append([key,[],[]])
        elif "score" in l or "dssp" in l:
            continue
        else:
            hi = l.split()[:5]+[0.0,0.0,0.0]
            hits[-1][1].append([int(hi[0]),int(hi[1])])
            hits[-1][2].append([float(hi[2]),float(hi[3]),float(hi[4])])

    # get per-hit statistics from an .hhr file
    # (!!! assume that .hhr and .atab have the same hits !!!)
    # [Probab, E-value, Score, Aligned_cols, 
    # Identities, Similarity, Sum_probs, Template_Neff]
    lines = open(infile[:-4]+'hhr', "r").readlines()
    pos = [i+1 for i,l in enumerate(lines) if l[0]=='>']
    for i,posi in enumerate(pos):
        hits[i].append([float(s) for s in re.sub('[=%]',' ',lines[posi]).split()[1::2]])
        
    # parse templates from FFDB
    for hi in hits:
        #if hi[0] not in ffids:
        #    continue
        entry = get_entry_by_name(hi[0], ffdb.index)
        if entry == None:
            continue
        data = read_entry_lines(entry, ffdb.data)
        hi += list(parse_pdb_lines(data))

    # process hits
    counter = 0
    xyz,qmap,mask,f0d,f1d,ids = [],[],[],[],[],[]
    for data in hits:
        if len(data)<7:
            continue
        
        qi,ti = np.array(data[1]).T
        _,sel1,sel2 = np.intersect1d(ti, data[6], return_indices=True)
        ncol = sel1.shape[0]
        if ncol < 10:
            continue
        
        ids.append(data[0])
        f0d.append(data[3])
        f1d.append(np.array(data[2])[sel1])
        xyz.append(data[4][sel2])
        mask.append(data[5][sel2])
        qmap.append(np.stack([qi[sel1]-1,[counter]*ncol],axis=-1))
        counter += 1

    xyz = np.vstack(xyz).astype(np.float32)
    mask = np.vstack(mask).astype(np.bool)
    qmap = np.vstack(qmap).astype(np.long)
    f0d = np.vstack(f0d).astype(np.float32)
    f1d = np.vstack(f1d).astype(np.float32)
    ids = ids
        
    return xyz,mask,qmap,f0d,f1d,ids

def load_ligand_from_pdb(fn, lig_name=None, remove_H=True):
    """Loads a small molecule ligand from pdb file `fn` into feature tensors.
    If no ligand is found, returns empty tensors with the same dimensions as
    usual.

    PDB format: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html

    Parameters
    ----------
    fn : str
        Name of PDB file
    lig_name : str
        3-letter residue name of ligand to load. If None, assumes
        there is only 1 ligand and loads it from all HETATM lines.
    remove_H : bool
        If True, does not load H atoms

    Returns
    -------
    xyz_sm : torch.Tensor (N_symmetry, L_sm, 3)
        Atom coordinates of ligand
    mask_sm : torch.Tensor (N_symmetry, L_sm)
        Boolean mask for whether atoms exist
    msa_sm : torch.Tensor (L_sm,)
        Integer-encoded (rf2aa.chemical) sequence (atom types) of ligand.
    bond_feats_sm : torch.Tensor (L_sm, L_sm)
        Bond features for ligand
    idx_sm : torch.Tensor (L_sm,)
        Residue number for ligand (all the same)
    atom_names : list of str
        Atom names of ligand (including whitespace) from columns 13-16 of
        PDB HETATM lines.
    """
    with open(fn, 'r') as fh:
        stream = [l for l in fh
                  if (("HETATM" in l) and (lig_name is None or l[17:20].strip()==lig_name))\
                     or "CONECT" in l]

    if len(stream)==0:
        sys.exit(f'ERROR (load_ligand_from_pdb): no HETATM records found in file {fn}.')

    mol, msa_sm, ins_sm, xyz_sm, mask_sm = \
        rf2aa.parsers.parse_mol("".join(stream), filetype="pdb", string=True, remove_H=remove_H,
                                find_automorphs=False)
    G = rf2aa.util.get_nxgraph(mol)
    bond_feats_sm = rf2aa.util.get_bond_feats(mol)

    atom_names = []
    for line in stream:
        if line.startswith('HETATM'):
            atom_names.append(line[12:16])

    return mol, xyz_sm, mask_sm, msa_sm, bond_feats_sm, atom_names


def read_multichain_template_pdb(Ls, templ_fn, align_conf=1.0, num_seq_classes=32):
    xyzs = dict()
    seqs = dict()
    all_chains = []
    Lasu = sum(Ls)

    with open(templ_fn) as fp:
        prior_resNo = None
        idx = 0
        for l in fp:
            if l[:4] != "ATOM":
                continue
            
            chn, resNo, atom, aa = l[21], int(l[22:26]), l[12:16], l[17:20]

            if prior_resNo is not None:
                if resNo != prior_resNo: idx += 1
            if (chn not in all_chains and (idx == 0 or idx == Lasu)):
                idx = 0
                xyzs[chn] = torch.full((Lasu, 27, 3), np.nan).float()
                seqs[chn] = torch.full((Lasu,), 20).long()
                all_chains.append(chn)

            chn = all_chains[-1]
            aa_idx = util.aa2num[aa] if aa in util.aa2num.keys() else 20

            for i_atm, tgtatm in enumerate(util.aa2long[aa_idx][:3]):
                if tgtatm == atom:
                    xyzs[chn][idx,i_atm,:] = torch.tensor([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                    break
            else:
                continue
            seqs[chn][idx] = aa_idx
            prior_resNo = resNo

    xyzs = torch.stack([xyzs[i] for i in all_chains]) # (N,Lasu,27,3)
    seqs = torch.stack([seqs[i] for i in all_chains]) # (N,Lasu)

    mask_t = torch.logical_not(torch.isnan(xyzs[...,0]))
    mask = mask_t[:,:,:3].all(dim=-1) 
    conf = torch.where(mask, torch.tensor(align_conf).float(), torch.zeros((len(all_chains),Lasu), dtype=xyzs.dtype))
    seqs = torch.nn.functional.one_hot(seqs, num_classes=num_seq_classes).float()
    t1d = torch.cat((seqs, conf[:,:,None]), -1)

    return xyzs, mask_t, t1d, seqs 