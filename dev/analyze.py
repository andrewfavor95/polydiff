import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RF2-allatom'))
import ipdb
import glob
import itertools
from dataclasses import dataclass
import os
import sys
from itertools import *
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import numpy as np
import parsers
import util

import sys
sys.path.append('/home/ahern/projects/pagan')
from swiss_army_knife import protein as sak
cmd = sak.get_cmd()
#
import estimate_likelihood as el
from inference import utils
from itertools import takewhile


#sns.set_theme()
#sns.set_style()

print('initializing analyze')
DESIGN = 'design_path'

def common_prefix(strlist):
    return ''.join(c[0] for c in takewhile(lambda x:
                all(x[0] == y for y in x), zip(*strlist)))


def read_metrics(df_path, add_contig_rmsd=True):
    df = pd.read_csv(df_path)
    df['method'] = 'placeholder_method'
    df['rundir'] = os.path.split(df_path)[0]
    df['run'] = df['name'].apply(lambda x: x.split('_')[0])
    # df = df[df['run'] =='run2'].reset_index()
    import re
    df['benchmark'] = [n[n.index('_')+1:n.index('_cond')] for n in df.name]

    df['run'] = [n[:n.index('_')] for n in df.name]
    # For backwards compatibility
    model_key = 'inference.ckpt_path'
    if model_key not in df.columns:
        model_key = 'inference.ckpt_override_path'

    models = df[model_key].unique()
    common = common_prefix(models)
    df['model'] = df[model_key].apply(lambda x: x[len(common):])
    #get_epoch  = lambda x: re.match('.*_(\w+).*', x).groups()[0]

    #df['model'] = df['inference.ckpt_path'].apply(get_epoch)
    # for tm_cluster in ['tm_cluster_0.40', 'tm_cluster_0.60', 'tm_cluster_0.80']:
    #     df['i_'+tm_cluster] = df.apply(lambda x: x[tm_cluster].split('_clus')[-1], axis=1)
    #df['contig_rmsd'] = df.apply(lambda x: get_contig_c_alpha_rmsd(x).item(), axis=1)
    return df

def combine(*df_paths, names=None):
    to_cat = []
    for i,p in enumerate(df_paths):
        df = read_metrics(p)
        #_, base = os.path.split(p)
        root, _ = os.path.splitext(p)
        root = root.split('/')[-2]
        df['source'] = names[i] if names else root
        to_cat.append(df)
    return pd.concat(to_cat)

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL', 'MAS'
    ]

aa2num= {x:i for i,x in enumerate(num2aa)}
aa2num['MEN'] = 20

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), #21 mask
]

def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        num = aa2num[aa]
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                info_het.append(dict(
                    idx=int(l[7:11]),
                    atom_id=l[12:16],
                    atom_type=l[77],
                    name=l[16:20]
                ))
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het

    return out


def get_input_pdb(row):
    trb = get_trb(row)
    # if args.template_dir is not None and os.path.exists(trbname):
    refpdb_fn = trb['config']['inference']['input_pdb'] # diffusion outputs
    return refpdb_fn

def pdb_to_xyz_idx(pdb, chain_i):
    parsed = parse_pdb(pdb)
    idxmap = dict(zip(parsed['pdb_idx'],range(len(parsed['pdb_idx']))))
    idx = [idxmap[e] for e in chain_i]
    return idx

def get_idx_motif(row, mpnn=True):
    rundir = row['rundir']
    trb = get_trb(row)
    # if args.template_dir is not None and os.path.exists(trbname):
    refpdb_fn = trb['config']['inference']['input_pdb'] # diffusion outputs
    pdb_ref = parse_pdb(refpdb_fn)
    if mpnn:
        pdb_des = parse_pdb(os.path.join(rundir, 'mpnn', row['name']+'_0.pdb'))
    else:
        pdb_des = parse_pdb(os.path.join(rundir, row['name']+'.pdb'))
    #pdb_des = parse_pdb(os.path.join(rundir, 'mpnn', row['name']+'.pdb'))
    # pdb_ref = parse_pdb(template_dir+trb['settings']['pdb'].split('/')[-1])
    xyz_ref = pdb_ref['xyz'][:,:3]

    # calculate 0-indexed motif residue positions (ignore the ones from the trb)
    # if os.path.exists(trbname):
    idxmap = dict(zip(pdb_ref['pdb_idx'],range(len(pdb_ref['pdb_idx']))))
    trb['con_ref_idx0'] = np.array([idxmap[i] for i in trb['con_ref_pdb_idx']])
    idxmap = dict(zip(pdb_des['pdb_idx'],range(len(pdb_des['pdb_idx']))))
    trb['con_hal_idx0'] = np.array([idxmap[i] for i in trb['con_hal_pdb_idx']])

    # calculate rmsds
    # row['rmsd_af2_des'] = calc_rmsd(xyz_pred.reshape(L*3,3), xyz_des.reshape(L*3,3))

    # load contig position
    # if os.path.exists(trbname): 
    idx_motif = [i for i,idx in zip(trb['con_hal_idx0'],trb['con_ref_pdb_idx']) 
                 if idx[0]!='R']

    L_motif = len(idx_motif)

    idx_motif_ref = [i for i,idx in zip(trb['con_ref_idx0'],trb['con_ref_pdb_idx']) 
                     if idx[0]!='R']
    xyz_ref_motif = xyz_ref[idx_motif_ref]
    # row['contig_rmsd_af2_des'] = calc_rmsd(xyz_pred[idx_motif].reshape(L_motif*3,3), 
    #                                        xyz_des[idx_motif].reshape(L_motif*3,3))
    # row['contig_rmsd_af2'] = calc_rmsd(xyz_pred[idx_motif].reshape(L_motif*3,3), xyz_ref_motif.reshape(L_motif*3,3))
    return idx_motif, idx_motif_ref

def get_trb(row):
    path = os.path.join(row['rundir'], f'{row["name"]}.trb')
    return np.load(path,allow_pickle=True)

def get_af2(row):
    rundir = row['rundir']
    d = rundir
    # if row['mpnn']:
    path = os.path.join(d, f'af2/{row["name"]}.pdb')
    if row.get('mpnn'):
        d = os.path.join(d, 'mpnn')
        path = os.path.join(d, f'af2/{row["name"]}_{row["mpnn_index"]}.pdb')
    return path

def load_af2(row, name=None):
    rundir = row['rundir']
    d = rundir
    # if row['mpnn']:
    path = os.path.join(d, f'af2/{row["name"]}.pdb')
    if row.get('mpnn'):
        d = os.path.join(d, 'mpnn')
        path = os.path.join(d, f'af2/{row["name"]}_{row["mpnn_index"]}.pdb')
    if row.get('ligmpnn'):
        d = os.path.join(d, 'ligmpnn')
        path = os.path.join(d, f'af2/{row["name"]}_{row["mpnn_index"]}.pdb')
    name = (name or row['model']) + '_af2'
    cmd.load(path, name)
    return name

def get_ligmpnn_path(row):
    rundir = row['rundir']
    return os.path.join(rundir, 'ligmpnn', f"{row['name']}_{row['mpnn_index']}.pdb")


def to_resi(chain_idx):
    return f'resi {"+".join(str(i) for _, i in chain_idx)}'

def to_chain(chain_idx):
    chains = set(ch for ch,_ in chain_idx)
    assert len(chains)==1
    return list(chains)[0]

def get_traj_path(row,  traj='X0'):
    rundir = row['rundir']
    if traj == 'X0':
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_pX0_traj.pdb')
    else:
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_Xt-1_traj.pdb')
    return traj_path

def load_traj(row, name=None, traj='X0'):
    rundir = row['rundir']
    if traj == 'X0':
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_pX0_traj.pdb')
    elif traj == 'Xt':
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_Xt-1_traj.pdb')
    else:
        trb = get_trb(row)
        if DESIGN in trb:
            traj_path = DESIGN
        else:
            #traj_path = os.path.join(row['rundir'], 'mpnn', row['name'] + '_'+str(row['mpnn_index'])+'.pdb')
            traj_path = os.path.join(row['rundir'], 'rethreaded', row['name'] + '_'+str(row['mpnn_index'])+'.pdb')
            if not os.path.exists(traj_path):
                traj_path = os.path.join(row['rundir'], row['name'] +'.pdb')
    name = name or row['model']
    cmd.load(traj_path, name)
    return name

from icecream import ic
def show_traj(row, strat, traj_type='X0'):
    strat_name = strat.replace(' ', '_')
    trb = get_trb(row)

    traj = load_traj(row, strat_name+'_'+traj_type, traj=traj_type)
    cmd.do('util.chainbow')
    color = 'white'
    if np.any(trb['con_hal_pdb_idx']):
        pymol_color(traj, trb['con_hal_pdb_idx'], color)
    only_backbones()


def show_traj_path(path):
    cmd.load(path, path.split('/')[-1])
    cmd.do('util.chainbow')
    only_backbones()
    
def pymol_color(name, chain_idx, color='red'):
    sel = f'{name} and resi {"+".join(str(i) for _, i in chain_idx)}'
    cmd.color(color, sel)

def get_motif_idx(row):
    #input_pdb = get_input_pdb(row)
    trb = get_trb(row)
    chain_idx = trb["con_hal_pdb_idx"]
    return torch.tensor([i for _, i in chain_idx])

def get_name(row, strat=None):
    strat = strat or row['name']
    strat_name = strat.replace(' ', '_')
    return strat_name

def to_selector(motif_resi):
    chains = set(ch for ch,_ in motif_resi)
    #if len(chains) == 1:
    #    return f'{name} and chain {to_chain(self.motif_resi)} and ({to_resi(self.motif_resi)})'
    chain_sels = []
    for ch, g in itertools.groupby(motif_resi, lambda x: x[0]):
        chain_sels.append(f'(chain {ch} and {to_resi(g)})')
    return ' or '.join(chain_sels)

@dataclass
class Structure:
    name: str
    motif_resi: any
    #diffusion_mask: torch.BoolTensor

    def motif_sele(self):
        return f'({self.name} and {to_selector(self.motif_resi)})'
        return to_selector(self.motif_resi)
        chains = set(ch for ch,_ in self.motif_resi)
        if len(chains) == 1:
            return f'{self.name} and chain {to_chain(self.motif_resi)} and ({to_resi(self.motif_resi)})'
        chain_sels = []
        for ch, g in itertools.groupby(self.motif_resi):
            chain_sels.append(f'(chain {ch} and resi {to_resi(g)})')
        return f"({self.name} and ({' or '.join(chain_sels)}))"

def load_motif_same_chain(row, strat=None):
    strat_name = get_name(row, strat)
    motif_suffix = '_motif'
    native = strat_name+motif_suffix
    input_pdb = get_input_pdb(row)
    trb = get_trb(row)
    ref_idx = trb["con_ref_pdb_idx"]
    if cmd.is_network:
        #cmd.do(f'load {input_pdb}, {native}; remove ({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
        cmd.load(f'{input_pdb}, {native}')
        cmd.remove(f'({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
    else:
        cmd.load(input_pdb, native)
        cmd.do(f'remove ({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
    for i, (chain, resi_i) in enumerate(ref_idx):
        cmd.alter(f'{native} and chain {chain} and resi {resi_i}', f'resi={i}')
        cmd.sort()
    return Structure(native, [(chi[0], i) for i, chi in enumerate(trb['con_ref_pdb_idx'])])
 
def load_motif(row, strat=None, show=True):
    strat_name = get_name(row, strat)
    motif_suffix = '_motif'
    native = strat_name+motif_suffix
    input_pdb = get_input_pdb(row)
    trb = get_trb(row)
    # ic(trb)
    # ligand_name = trb['inference.ligand']
    ligand_name = row['inference.ligand']
    ref_idx = trb["con_ref_pdb_idx"]


    #cmd.do(f'load {input_pdb}, {native}; remove ({native} and not ({to_selector(ref_idx)} or resn {row["inference.ligand"]}))')
    cmd.load(f'{input_pdb}', f'{native}')
    cmd.remove(f'({native} and not ({to_selector(ref_idx)} or resn {ligand_name}))')
    #cmd.remove(f'({native} and not ({to_selector(ref_idx)}))')
    # cmd.do(f'load {input_pdb}, {native}; remove ({native} and not {to_selector(ref_idx)})')
    # cmd.load(input_pdb, native)
    #ic(f'remove ({native} and not ({to_selector(ref_idx)}))')
    #ipdb.set_trace()
    #cmd.do(f'remove ({native} and not ({to_selector(ref_idx)}))')
    #ipdb.set_trace()
    cmd.show_as('licorice', native)
    #ipdb.set_trace()
    #print(f'load {input_pdb}, {native}; remove ({native} and not ({to_selector(ref_idx)})')
    #for i, (chain, resi_i) in enumerate(ref_idx):
    #    cmd.alter(f'{native} and chain {chain} and resi {resi_i}', f'resi={i}')
    #for i, (chain, resi_i) in enumerate(ref_idx):
    #    cmd.alter(f'{native}', f'chain={chain}')
    #for i, (chain, resi_i) in enumerate(ref_idx):
    #    cmd.alter(f'{native} and resi {resi_i}', f'resi={i}')
    #cmd.sort()

    # for i, (chain, resi_i) in enumerate(ref_idx):
    #     cmd.alter(f'{native}', f'chain="A"')
    for i, (chain, resi_i) in enumerate(ref_idx):
        cmd.alter(f'{native} and resi {resi_i}', f'resi={i}')
    cmd.sort()
    return Structure(native, [(ch, i) for i, (ch, _) in enumerate(trb['con_ref_pdb_idx'])])
    # return Structure(native, [('A', i) for i, chi in enumerate(trb['con_ref_pdb_idx'])])
    #return Structure(native, [(chi[0], i) for i, chi in enumerate(trb['con_ref_pdb_idx'])])
    #return Structure(native, trb['con_ref_pdb_idx'])

def show_motif(row, strat, traj_types='X0', show_af2=True, show_true=False):
    structures = {}
    rundir = row['rundir']
    strat_name = strat.replace(' ', '_')
    trb = get_trb(row)
    native = load_motif(row, strat_name)
    structures['native'] = native
    
#     input_pdb = get_input_pdb(row)
#     motif_suffix = '_motif'
#     native = strat_name+motif_suffix
#     ic(input_pdb)
#     # cmd.load(input_pdb, native)
#     ref_idx = trb["con_ref_pdb_idx"]
#     # cmd.do(f'load {input_pdb}, {native}; remove not (chain {to_chain(ref_idx)} and {to_resi(ref_idx)})')
#     # cmd.do(f'load {input_pdb}, {native}; remove not ({native} and chain {to_chain(ref_idx)} and {to_resi(ref_idx)})')
#     # cmd.do(f'load {input_pdb}, {native}')
#     # cmd.do(f'load {input_pdb}, {native}; remove ({native} and chain {to_chain(ref_idx)} and not ({to_resi(ref_idx)}))')
#     cmd.do(f'load {input_pdb}, {native}; remove ({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
#     cmd.load(input_pdb, native)
#     cmd.do(f'remove ({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
    # return
    
    if not isinstance(traj_types, list):
        traj_types = [traj_types]
    trajs = []
    traj_motifs = []
    structures['trajs'] = []
    for traj_type in traj_types:
        traj = load_traj(row, strat_name+'_'+traj_type, traj_type)
        trajs.append(traj)
        traj_motif = f'{traj} and {to_resi(trb["con_hal_pdb_idx"])}'
        traj_motifs.append(traj_motif)
        #ic(traj_motif, native)
        # cmd.align(traj_motif, native, 'mobile_state=1')
        cmd.align(traj_motif, native.name)
        structures['trajs'].append(Structure(traj, trb['con_hal_pdb_idx']))
    # traj = strat_name
    # traj_path = os.path.join(rundir, 'mpnn', row['name'])+'.pdb'
    # # print(traj_path)
    # cmd.load(os.path.join(rundir, 'mpnn', row['name'])+'.pdb', strat_name)
    # cmd.show_as('
    #native_motif = f'{traj} and {to_resi(trb["con_ref_pdb_idx"])}'
    # cmd.do(f'load {input_pdb} {native}; remove not ({native_motif})')
    # des_motif_resi = to_resi(trb["con_hal_pdb_idx"])
    # cmd.align(af2, traj)
    # cmd.align(native , traj)
    #cmd.align(traj, native)
    
    if show_af2:
        af2 = load_af2(row, strat_name)
        #cmd.align(af2, native)
        #cmd.align(af2, traj)
        af2 = Structure(af2, trb['con_hal_pdb_idx'])
        structures['af2'] = af2
        #ic(af2.motif_sele(), native.motif_sele())
        cmd.align(af2.motif_sele(), native.motif_sele())
        cmd.set('stick_transparency', 0.7, af2.name)
    #cmd.center(traj)
    cmd.center(native.motif_sele())
    cmd.do('util.chainbow')
    color = 'white'
    # pymol_color(native, trb['con_ref_pdb_idx'], color)
    for traj in trajs:
        pymol_color(traj, trb['con_hal_pdb_idx'], color)
    if show_af2:
        pymol_color(af2.name, trb['con_hal_pdb_idx'], color)
    whole_native = None
    if show_true:
        #input_pdb = get_input_pdb(row)
        whole_native = strat_name+'_true'
        ref_idx = trb["con_ref_pdb_idx"]
        #cmd.do(f'load {input_pdb}, {whole_native}')
        cmd.load(input_pdb, whole_native)
        pymol_color(whole_native, ref_idx, color)
        
        
    #cmd.color('pink', '*'+motif_suffix)
    #cmd.color('pink', native.name)
    cmd.color('pink', f'{native.name} and elem C')
    # cmd.hide('cartoon', native)
    # cmd.show('cartoon', f'{native} and {to_resi(trb["con_ref_pdb_idx"])}')
    only_backbones()
    return structures
    #return ructure(t,[]) for t,m in zip(trajs, traj_motifs)], 
    #return trajs, traj_motifs, whole_native, native, native

def only_backbones():
    cmd.hide('all')
    cmd.show('licorice', '(name ca or name c or name n)')


def get_spread(bench_des, key='contig_rmsd_af2'):
    bench_des.reset_index()
    bench_des = bench_des.sort_values(key)
    worst = bench_des.iloc[-1]
    best = bench_des.iloc[0]
    median = bench_des.iloc[len(bench_des)//2]
    rows = [best, median, worst]
    return rows

def show_spread(bench_des, key='contig_rmsd_af2'):
    sak.clear(cmd)
    rows = get_spread(bench_des, key=key)
    # rows = rows[:2]
    for row in rows:
        name = f"ep{row['model']}_{row[key]:.2f}"
        show_motif(row, name)
        # break
        

def get_examples(df, benchmark, model, mpnn=False, key='rmsd_af2_des'):
    bench_des = df[(df['benchmark'] == benchmark) & (df['model'] == model) & (df['mpnn'] == mpnn)]
    bench_des.reset_index()
    return get_spread(bench_des, key)

def get_traj_xyz(row, traj_type='X0', n=None):
    #p = parse_pdb(os.path.join(row['rundir'], 'mpnn', row['name']+'.pdb'))

    motif_idx = get_motif_idx(row)

    traj_path = get_traj_path(row, traj_type)
    return read_traj_xyz(traj_path)

def read_traj_xyz(traj_path, seq=False):
    with open(traj_path) as f:
        s = f.read()
        models = s.strip().split('ENDMDL')
        parsed = []
        seqs = []
        for i, m in enumerate(models):
            if not m:
                continue
            # o = parsers.parse_pdb_lines(m.split('\n'), False, False, lambda x: 0)
            o = parsers.parse_pdb_lines(m.split('\n'), False, seq, lambda x: 0)
            xyz = o[0]
            if seq:
                seqs.append(o[-1])
            parsed.append(xyz)
        # parsed = torch.concat(parsed)
    parsed = torch.tensor(np.array(parsed))
    if seqs:
        return parsed, seqs
    return parsed
        # parsed = [parse_pdb_lines(m.split('\n')) for m in models]

def get_contig_c_alpha_rmsd(row, all_idxs=False):
    print('.', end='')
    traj = get_traj_xyz(row, 'Xt')
    motif_idx, native_motif_idx = get_idx_motif(row, mpnn=False)
    i = torch.tensor([0,1])
    if all_idxs:
        #i = torch.arange(traj.shape[1])
        ic(traj.shape)
        i = torch.arange(traj.shape[0])
    return el.c_alpha_rmsd_traj(traj[i][:,motif_idx])

def make_row_from_traj(traj_prefix):
    synth_row = {}
    
    synth_row['rundir'], synth_row['name'] = os.path.split(traj_prefix)
    synth_row['mpnn_index'] = 0
    synth_row['mpnn'] = True
    trb = get_trb(synth_row)
    # synth_row.update(trb['config'])
    for k,vdict in trb['config'].items():
        for k2,v in vdict.items():
            synth_row[k+'.'+k2] = v
    return synth_row
    #show_motif(synth_row, 'synth_row_2', show_af2=False)

def show_row(row, traj_name, traj_type='X0'):
    show_traj(row, traj_name, traj_type)
    #input_pdb = get_input_pdb(row)
    
    af2 = load_af2(row, traj_name)
    cmd.align(af2, traj_name)
    only_backbones()
    cmd.do('util.chainbow')
    cmd.set('stick_transparency', 0.7, af2)
    # break

def calc_rmsd(xyz1, xyz2, eps=1e-6):
    #ic(xyz1.shape, xyz2.shape)

    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    L = xyz2_.shape[0]
    rmsd = np.sqrt(np.sum((xyz2_-xyz1)*(xyz2_-xyz1), axis=(0,1)) / L + eps)

    return rmsd


import os
import shutil
from itertools import permutations
from collections import defaultdict

def get_benchmark(spec, interc=10-100, length=150):
	'''
	Spec like:
		{'GAA_0_0': {
			'pdb': '/home/heisen/0_projects/4_enzymedesign/input/GAA/5nn5_aligned_sub_rot1.pdb',
		  	'ligand_resn': 'UNL',
		  	'pdb_contig': [('A', 518), ('A', 616), ('A', 404)]
			},
		}

	'''
	input_dir = '/home/ahern/projects/clean_rf_diffusion/benchmark/input'
	theo_dir = 'enzyme'
	benchmarks_json = '''
	{
	'''

	benchmark_dict = OrderedDict()
	for name, d in spec.items():
		for motif in permutations(row.motif_selector):
			motif_suffix = ''.join(f'{ch}{i}' for ch,i in motif)
			benchmark_name = f'{input_pdb}_{motif_suffix}'
			# bench_by_pdb[row.pdb].append(
			contig = interc + ',' + (','+interc+',').join(f'{ch}{r}-{r}' for ch, r in motif) +',' + interc
			benchmark_dict[benchmark_name] = f"inference.input_pdb={d['pdb']} contigmap.contigs=[\\\\'{contig}\\\\']"
	benchmarks_json = json.dumps(benchmark_dict, indent=4)
	return benchmarks_json
	# print(benchmarks_json)

def make_script(args, run, debug=False, n_sample=1, num_per_condition=1):
    T_arg = ''
    seq_per_target = 8
    lengths = '150-150|200-200'
    if debug:
        n_sample = 1
        T_arg = 'diffuser.T=5 '
        seq_per_target = 1
        lengths = '150-150'
        num_per_condition = 1
    
    arg_strs = []
    for a in args:
        arg_strs.append(f"""        "--config-name=base inference.deterministic=False inference.align_motif=True inference.annotate_termini=True inference.model_runner=NRBStyleSelfCond inference.ckpt_path=/home/ahern/projects/rf_diffusion/models/theo_pdb/BFF_4.pt contigmap.length={lengths} {T_arg}{a}" \\""")

    nl = '\n'
    script = f"""#!/bin/bash 

source activate /home/dimaio/.conda/envs/SE3nv 

./pipeline.py \\
        --num_per_condition {num_per_condition} --num_per_job 1 --out {run}/run \\
        --args \\
{nl.join(arg_strs)}
        --num_seq_per_target {seq_per_target} --af2_gres=gpu:a6000:1 -p cpu
    """
    return script

import functools

def add_filters(df, **kwargs):
    filter_names, filter_unions = add_filters_multi(df, **kwargs)
    return filter_names + filter_unions

def add_filters_multi(df, threshold_columns = ['contig_rmsd_af2', 'rmsd_af2_des'], threshold_signs = ['-', '-'], thresholds = [(1, 3)]):
    filter_names = []
    filter_unions = []
    for i, threshold in enumerate(thresholds):
        filters = []
        filter_names = []
        for metric, sign, value in zip(threshold_columns, threshold_signs, threshold):
            if sign == '+':
                geq_or_leq = '>'
                passes_filter = df[metric] > value
            else:
                geq_or_leq = '<'
                passes_filter = df[metric] < value
            filter_name = f'{metric}_{geq_or_leq}_{value}'
            filter_names.append(filter_name)
            df[filter_name] = passes_filter
        filter_union_name = f'filter_set_{i}'
        filter_unions.append(filter_union_name)
        # df[functools.reduce(lambda a,b: df[a] & df[b], filter_names)]
        #df[filter_union_name] = functools.reduce(lambda a,b: df[a] & df[b], filter_names)
        df[filter_union_name] = functools.reduce(lambda a,b: a & b, [df[f] for f in filter_names])
    return filter_names, filter_unions

def melt_filters(df, filter_names):
    data = df.melt(id_vars='name', value_vars=filter_names, var_name='filter_name', value_name='pass')
    merged=data.merge(right=df, on='name', how='outer')
    return merged

def plot_melted(df, filter_names):
    hue_order = sorted(df['contigmap.length'].unique())
    x_order = sorted(df['benchmark'].unique())
    g = sns.catplot(data=df, y='pass', hue='contigmap.length', x='benchmark', kind='bar', orient='v', col='filter_name', hue_order=hue_order, height=8.27, aspect=11.7/8.27, legend_out=True, order=x_order, ci=None)
    # iterate through axes
    for ax in g.axes.ravel():

        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()*100):.1f}%' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)
        ax.tick_params(axis='x', rotation=90)
    plt.xticks(rotation=90)
    return g
    # filter_union = functools.reduce(lambda a,b: a | b, filters)

def plot_melted_by_column(df, filter_names, column):
    for column_v in df[column].unique():
        g = plot_melted(df[df[column]==column_v], filter_names)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f'{column}:{column_v}')

def add_benchmark_from_input_pdb(df):
    def get_benchmark(row):
            return os.path.basename(os.path.splitext(row['inference.input_pdb'])[0])
    df['benchmark'] = df.apply(get_benchmark, axis=1)

def get_lengths(row):
    m = analyze.get_trb(row)['sampled_mask'][0]
    l = [e for e in m.split(',') if e[0] != 'A']
    l = [int(e.split('-')[0]) for e in l]
    return l

def get_sidechain_rmsd(row):
    af2_motif = get_af2_motif(row)
    native_motif = get_native_motif(row)
    is_atom = get_motif_atom_mask(row)
    is_atom = is_atom[:,:14]
    return analyze.calc_rmsd(af2_motif[is_atom].numpy(), native_motif[is_atom].numpy())

def get_motif_from_pdb(pdb, chain_i):
    feats = utils.process_target(pdb, parse_hetatom=True, center=False)
    xyz_motif_idx = pdb_to_xyz_idx(pdb, chain_i)
    motif = feats['xyz_27'][xyz_motif_idx]
    return motif

def get_min_dist_to_het(pdb, chain_i):
    feats = utils.process_target(pdb, parse_hetatom=True, center=False)
    xyz_motif_idx = pdb_to_xyz_idx(pdb, chain_i)
    motif = feats['xyz_27'][xyz_motif_idx]
    motif_atms = motif[feats['mask_27'][xyz_motif_idx]]
    het_names = np.array([i['name'].strip() for i in feats['info_het']])
    het_xyz = feats['xyz_het'][het_names != 'HOH']
    het_names =  het_names[het_names != 'HOH']
    if het_xyz.size == 0:
        return 999.0, ''
    try:
        dgram = torch.cdist(motif_atms[None], torch.tensor(het_xyz[None], dtype=torch.float32))
    except Exception as e:
        print(motif_atms.shape, het_xyz.shape, len(het_xyz))
        raise e
    # print(dgram.shape)
    # print(set(het_names))
    minidx  = torch.argmin(dgram, keepdim=True)
    _, M, H = dgram.shape
    hetidx = minidx // M
    return torch.min(dgram), het_names[hetidx]
    print(f'{torch.min(dgram)} to {het_names[hetidx]}')

def get_traj_motif(row):
    motif_idx, native_motif_idx = analyze.get_idx_motif(row, mpnn=False)
    traj = analyze.get_traj_xyz(row, 'X0')
    return traj[0, motif_idx]

def get_design_pdb(row):
    rundir = row['rundir'] 
    return os.path.join(rundir, row['name']+'.pdb')

def get_design(row):
    rundir = row['rundir'] 
    des =  utils.process_target(os.path.join(rundir, row['name']+'.pdb'))
    return des['xyz_27']

def get_af2_xyz(row):
    af2_path = get_af2(row)
    des =  utils.process_target(af2_path)
    return des['xyz_27']


# def get_af2_motif(row):
#     motif_idx, native_motif_idx = analyze.get_idx_motif(row, mpnn=False)
#     af2 = analyze.get_af2(row)
#     af2 = utils.process_target(af2)
#     return af2['xyz_27'][motif_idx][:,:14]

def get_native(row):
    # motif_idx, native_motif_idx = analyze.get_idx_motif(row, mpnn=False)
    input_pdb = get_input_pdb(row)
    native = utils.process_target(input_pdb, center=False, parse_hetatom=True)
    #print(native.keys())
    het_names = set([i['name'].strip() for i in native['info_het']])
    assert len(het_names) == 1, f'more than 1 het: {het_names}'
    return native['xyz_27'][:,:14], native['xyz_het']
    # return native['xyz_27'][native_motif_idx][:,:14]

def get_registered_ligand(row, af2=False):
    motif_idx, native_motif_idx = get_idx_motif(row, mpnn=False)
    # traj_motif = get_traj_motif(row)
    if af2:
        des = get_af2_xyz(row)
    else:
        des = get_design(row)
    native, het = get_native(row)
    motif_des = des[motif_idx]
    motif_native = native[native_motif_idx]
    T = sak.register_full_atom(motif_des[:,1:2,:], motif_native[:,1:2,:])
    des = T(des)
    return des, native, het

def get_dist_to_ligand(row, af2=False, c_alpha=False):
    des, native, het = get_registered_ligand(row, af2=af2)
    motif_idx, native_motif_idx = get_idx_motif(row, mpnn=False)
    L, _, _ = des.shape
    if c_alpha:
        bb_des = des[:,1]
    else:
        bb_des = des[:,:3].reshape(L*3, 3)
    dgram = torch.cdist(bb_des[None,...], torch.tensor(het[None, ...], dtype=torch.float32), p=2)
    return dgram[0]

null_structure = Structure('null', [])

def show_motif_simple(row, strat, traj_types='X0', show_af2=True, show_true=False):
    structures = {}
    rundir = row['rundir']
    strat_name = strat.replace(' ', '_')
    trb = get_trb(row)
    has_motif = bool(trb['con_hal_pdb_idx'])
    # native=null_structure
    native=None
    if has_motif:
        native = load_motif(row, strat_name)
    structures['native'] = native
    
    if not isinstance(traj_types, list):
        traj_types = [traj_types]
    trajs = []
    traj_motifs = []
    structures['trajs'] = []
    for traj_type in traj_types:
        traj = load_traj(row, strat_name+'_'+traj_type, traj_type)
        trajs.append(traj)
        traj_motif = f'{traj} and {to_resi(trb["con_hal_pdb_idx"])}'
        traj_motifs.append(traj_motif)
        if has_motif:
            cmd.align(traj_motif, native.name)
        structures['trajs'].append(Structure(traj, trb['con_hal_pdb_idx']))
    
    if show_af2:
        af2 = load_af2(row, strat_name)
        af2 = Structure(af2, trb['con_hal_pdb_idx'])
        structures['af2'] = af2
        if has_motif:
            cmd.align(af2.motif_sele(), native.motif_sele())
        cmd.set('stick_transparency', 0.7, af2.name)
    color = 'white'
    for traj in trajs:
        if has_motif:
            pymol_color(traj, trb['con_hal_pdb_idx'], color)
    if show_af2 and has_motif:
        pymol_color(af2.name, trb['con_hal_pdb_idx'], color)
    whole_native = None
    if show_true:
        whole_native = strat_name+'_true'
        ref_idx = trb["con_ref_pdb_idx"]
        cmd.load(input_pdb, whole_native)
        pymol_color(whole_native, ref_idx, color)
    
    #if not has_motif:
    #    cmd.hide('everything', native.name)
    if has_motif:
        cmd.color('pink', f'{native.name} and elem C')
    return structures

def show_paper_pocket(row, des=True, ligand=False):
    # b=row['benchmark']
    b = row['name']
    structures = show_motif_simple(row, b, traj_types=['des'])
    # structures = analyze.show_motif(row, b, traj_types=['X0'])
    af2 = structures['af2']
    des = structures['trajs'][0]
    native = structures['native']
    af2_scaffold = Structure(af2.name +'_scaffold', af2.motif_resi)
    cmd.copy(af2_scaffold.name, af2.name)

    cmd.do(f'util.cbag {af2.name}')
    cmd.do(f'util.cbag {des.name}')
    cmd.hide('everything', f'{af2_scaffold.name} or {af2.name} or {native.name} or {des.name}')
    # Scaffold
    cmd.show('cartoon', f'{af2_scaffold.name}')
    cmd.set('stick_transparency', 0, af2.name)
    cmd.color('gray', f'{af2_scaffold.name}')
    cmd.color('teal', f'{af2.motif_sele()} and elem C and not (name ca or name c or name n)')
    
    # Design
    if des:
        #cmd.align(des.name, af2_scaffold.name)
        cmd.align(f'{des.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
        cmd.show('cartoon', f'{des.name}')
        cmd.do(f'mass_paper_rainbow_sel {des.name}')

    # AF2 sidechains
    cmd.show('licorice', f'({af2.motif_sele()}) and not (name o)')
    cmd.color('paper_pink', f'({af2.motif_sele()}) and (elem C or name n)')

    # Desired sidechains
    cmd.show('licorice', f'{native.name} and not (name o)')
    cmd.color('paper_teal', f'{native.name} and (elem C or name n)')

    cmd.set('cartoon_transparency', 0)
    cmd.center(af2.name)
    cmd.hide('everything', af2.name)
    cmd.hide('everything', af2_scaffold.name)
    cmd.hide('cartoon', des.name)
    cmd.show('licorice', f'{des.name} and (name c or name ca or name n)')
    cmd.set('stick_transparency', 0, des.name)
    # cmd.show('licorice'
    # cmd.show('licorice', f'{des.motif_sele()}')
    
    # #
    # cmd.align(f'{af2.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
    # cmd.show('licorice', f'{af2.name} and (name c or name ca or name n)')
    # cmd.show('licorice', f'{af2.motif_sele()}')
    # cmd.color('good_gray', f'{af2.name} and (name c or name ca or name n)')
    
    # Ligand
    lig = f'lig_{b}'
    if ligand:
        cmd.load(get_input_pdb(row), lig)
        cmd.color('orange', f'{lig} and elem C')
    # cmd.orient(lig)
    return [af2.name, af2_scaffold.name, des.name, native.name, lig]

def show_paper_pocket_af2(row, b=None, des=True, ligand=False, traj_types=None, show_af2=True):
    # b=row['benchmark']
    b = b or f"{row['name']}_{row['mpnn_index']}"
    traj_types = traj_types or ['des']
    structures = show_motif_simple(row, b, traj_types=traj_types, show_af2=show_af2)
    has_motif = structures['native'] is not None
    # structures = analyze.show_motif(row, b, traj_types=['X0'])
    des = structures['trajs'][0]
    native = structures['native']
    af2 = None
    af2_scaffold=None
    if show_af2:
        af2 = structures['af2']
        af2_scaffold = Structure(af2.name +'_scaffold', af2.motif_resi)
        if has_motif:
            cmd.align(f'{af2.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
        else:
            cmd.align(f'{des.name} and name ca', f'{af2.name} and name ca')
            # import ipdb
            # ipdb.set_trace()
        # else:
        #     cmd.align(f'{af2.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')

        cmd.copy(af2_scaffold.name, af2.name)

        cmd.do(f'util.cbag {af2.name}')
    cmd.do(f'util.cbag {des.name}')
    if has_motif:
        cmd.hide('everything', f'{native.name}')
    cmd.hide('everything', f'{des.name}')
    if show_af2:
        cmd.hide('everything', f'{af2_scaffold.name} or {af2.name}')
        # Scaffold
        cmd.show('cartoon', f'{af2_scaffold.name}')
        cmd.set('stick_transparency', 0, af2.name)
        cmd.color('good_gray', f'{af2_scaffold.name}')
    
    # Design
    if des:
        #cmd.align(des.name, af2_scaffold.name)
        if show_af2:
            cmd.align(f'{des.name}', f'{af2.name}')
        if has_motif:
            cmd.align(f'{des.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
        cmd.show('cartoon', f'{des.name}')
        cmd.do(f'mass_paper_rainbow_sel {des.name}')

    if show_af2 and has_motif:
        # AF2 sidechains
        cmd.show('licorice', f'({af2.motif_sele()}) and not (name o)')
        cmd.color('good_gray', f'({af2.motif_sele()}) and (elem C or name n)')

    # Desired sidechains
    if has_motif:
        cmd.show('licorice', f'{native.name} and not (name o)')
        cmd.color('paper_teal', f'{native.name} and (elem C or name n)')
        cmd.hide('everything', f'{native.name} and not {native.motif_sele()}')
        # trb = get_trb(row)
        cmd.show('licorice', f'{native.name} and resn {row["inference.ligand"]}')
        cmd.color('orange', f'{native.name} and resn {row["inference.ligand"]} and elem C')

    cmd.set('cartoon_transparency', 0)
    if show_af2:
        cmd.center(af2.name)
    else:
        cmd.center(des.name)
    if has_motif:
        cmd.hide('sticks', f'{native.name} and elem H')
    #cmd.hide('everything', af2.name)
    #cmd.hide('everything', af2_scaffold.name)
    #cmd.hide('cartoon', des.name)
    #cmd.show('licorice', f'{des.name} and (name c or name ca or name n)')
    #cmd.set('stick_transparency', 0, des.name)
    # cmd.show('licorice'
    # cmd.show('licorice', f'{des.motif_sele()}')
    
    # #
    #cmd.align(f'{af2.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
    # cmd.show('licorice', f'{af2.name} and (name c or name ca or name n)')
    # cmd.show('licorice', f'{af2.motif_sele()}')
    # cmd.color('good_gray', f'{af2.name} and (name c or name ca or name n)')
    
    # Ligand
    lig = f'lig_{b}'
    if ligand:
        cmd.load(get_input_pdb(row), lig)
        cmd.color('orange', f'{lig} and elem C')
    # cmd.orient(lig)
    identifiers =  [af2, af2_scaffold, des, native]
    if ligand:
        identifiers.append(Structure(lig, None))
    return identifiers



def plot_success(df, threshold_columns=['contig_rmsd_af2', 'rmsd_af2_des', 'af2_pae_mean'], threshold_signs=['-', '-', '-'], named_thresholds=[('excellent', (1,2,5)), ('good', (1.5,3,7.5)), ('okay', (2,3,10))], recompute=True):
    #filters,  filter_unions = analyze.add_filters_multi(df, threshold_columns=['contig_rmsd_af2','contig_rmsd_af2_full_atom', 'rmsd_af2_des', 'af2_pae_mean'], thresholds=[(1,1.5,2,5), (1,999,2,5), (1.5,3,3,7.5), (1.5,2,3,10)], threshold_signs=['-','-', '-', '-'])
    # filter_names = ['excellent', 'backbone excellent', 'good', 'okay']
    filter_names = [name for name, threshold in named_thresholds]
    thresholds = [threshold for name, threshold in named_thresholds]
    if recompute:
        filters,  filter_unions = add_filters_multi(df, threshold_columns=threshold_columns, thresholds=thresholds, threshold_signs=threshold_signs)
        df.drop(columns=filter_names, inplace=True, errors='ignore')
        df.rename(columns=dict(zip(filter_unions, filter_names)), inplace=True)
    else: 
        filter_unions = []
        for i in range(len(named_thresholds)):
            filter_union_name = f'filter_set_{i}'
            filter_unions.append(filter_union_name)

    melts = []
    for filter_union in filter_names:
        best_filter_passers = df.groupby(["name"]).apply(lambda grp: grp.sort_values([filter_union, 'contig_rmsd_af2_full_atom'], ascending=[False, True]).head(1))
        best_filter_passers.index =best_filter_passers.index.droplevel()
    # best_filter_passers = df.loc[df.groupby(["name"])["contig_rmsd_af2"].idxmin()]
    # filters = analyze.add_filters(best_filter_passers, thresholds=[(1,2)])
        melted = melt_filters(best_filter_passers, [filter_union])
        melted['filter_set'] = filter_union
        melts.append(melted)

    melted = pd.concat(melts)
    # melted['filter_name'] = melted['filter_name'].replace(filter_unions, filter_names)
    # analyze.plot_melted(melted, filter_unions)

    # hue_order = sorted(df['contigmap.length'].unique())
    import matplotlib.pyplot as plt
    x_order = sorted(df['benchmark'].unique())
    g = sns.catplot(data=melted, y='pass', hue='filter_name', x='benchmark', kind='bar', orient='v', height=8.27, aspect=11.7/8.27, legend_out=True, order=x_order, ci=None)
    # iterate through axes
    for ax in g.axes.ravel():

        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()*100):.1f}%' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)
        ax.tick_params(axis='x', rotation=90)
    plt.xticks(rotation=90)
    return g, melted

from tqdm.notebook import trange, tqdm
def apply(df, name, f):
    df = df.copy()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        df.loc[i, name] = f(row)
    return df

def apply_arr(df, name, f):
    df = df.copy()
    kw = {name: None}
    df = df.assign(**kw)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        df.at[i, name] = f(row)
    return df


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def add_ligand_dist(df, c_alpha=False):
    groupers = ['name']
    if 'source' in df.columns:
        groupers = ['source'] + groupers
    designs = df.drop_duplicates(groupers, ignore_index=True)
    name = ('c_alpha' if c_alpha else 'bb') + '_ligand_dist'
    designs = apply_arr(designs, name, lambda x: get_dist_to_ligand(x, c_alpha=c_alpha).min(-1)[0].numpy())
    return df.merge(designs[groupers + [name]], on=groupers, how='inner')



def show_df(data, cols=['af2_pae_mean', 'rmsd_af2_des'], n=999):
    i=1
    for _, row in itertools.islice(data.iterrows(), n):
        rmsd_too_high = row['rmsd_af2_des'] > 2
        pae_too_high =  row['af2_pae_mean'] > 5
        sc = not rmsd_too_high and not pae_too_high
        key_val = [f'i_{i}']
        for k in cols:
            v = row[k]
            if not isinstance(v, str):
                v = f'{v:.1f}'
            key_val.append(f'{k}_{v}')
            print(key_val)
        design_name = '__'.join(key_val)
        print(design_name)
        structures = show_paper_pocket_af2(row, design_name)
        for s in structures:
            if s:
                cmd.set('grid_slot', i, s.name)
        i += 1
        af2, af2_scaffold, des, motif = structures
        cmd.super(f'{des.name} and name ca', f'{af2.name} and name ca')
        if rmsd_too_high and pae_too_high:
            cmd.color('purple', af2_scaffold.name)
        elif rmsd_too_high: 
            cmd.color('red', af2_scaffold.name)
        elif pae_too_high: 
            cmd.color('blue', af2_scaffold.name)
    cmd.set('grid_mode', 1)