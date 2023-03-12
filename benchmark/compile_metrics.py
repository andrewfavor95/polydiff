#!/usr/bin/env python
#
# Compiles metrics from scoring runs into a single dataframe CSV
#

import os, argparse, glob
import numpy as np
import pandas as pd
from icecream import ic
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs')
    parser.add_argument('--outcsv',type=str,default='compiled_metrics.csv',help='Output filename')
    args = parser.parse_args()

    filenames = glob.glob(args.datadir+'/*.trb')

    print('loading run metadata (base metrics)')
    records = []
    for fn in tqdm(filenames):
        name = os.path.basename(fn).replace('.trb','')
        trb = np.load(fn, allow_pickle=True)

        record = {'name':name}
        if 'lddt' in trb:
            record['lddt'] = trb['lddt'].mean()
        if 'inpaint_lddt' in trb:
            record['inpaint_lddt'] = np.mean(trb['inpaint_lddt'])

        if 'plddt' in trb:
            plddt = trb['plddt'].mean(1)
            record.update(dict(
                plddt_start = plddt[0],
                plddt_mid = plddt[len(plddt)//2],
                plddt_end = plddt[-1],
                plddt_mean = plddt.mean()
            ))
        if 'sampled_mask' in trb:
            record['sampled_mask'] = trb['sampled_mask']
        if 'config' in trb:
            for k,vdict in trb['config'].items():
                for k2,v in vdict.items():
                    record[k+'.'+k2] = v
        records.append(record)

    df_base = pd.DataFrame.from_records(records)

    # load computed metrics, if they exist
    print('loading computed metrics')
    # accumulate metrics for: no mpnn, mpnn, ligand mpnn
    df_all_list = []

    # metrics of "no mpnn" designs
    df_nompnn = df_base.copy()
    for path in [
        args.datadir+'/af2_metrics.csv.*',
        args.datadir+'/pyrosetta_metrics.csv.*',
    ]:
        df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        df_nompnn = df_nompnn.merge(tmp, on='name', how='outer')

    if df_nompnn.shape[1] > df_base.shape[1]: # were there designs that we added metrics for?
        df_nompnn['mpnn'] = False
        df_nompnn['ligmpnn'] = False
        df_all_list.append(df_nompnn)

    # MPNN and LigandMPNN metrics
    def _load_mpnn_df(mpnn_dir, df_base):
        df_accum = pd.DataFrame(dict(name=[]))
        for path in [
            mpnn_dir+'/af2_metrics.csv.*',
            mpnn_dir+'/pyrosetta_metrics.csv.*',
        ]:
            df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
            tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
            n_unique_names = len(set(tmp['name']))
            n_names = len(tmp)
            if n_unique_names < n_names:
                print('Dropping {n_names - n_unique_names}/{n_names} duplicates from {path}')
                tmp.drop_duplicates('name', inplace=True)
            df_accum = df_accum.merge(tmp, on='name', how='outer')

        # chemnet
        chemnet_dfs = [pd.read_csv(fn,index_col=None) for fn in glob.glob(mpnn_dir+'/chemnet_scores.csv.*')]
        tmp = pd.concat(chemnet_dfs) if len(chemnet_dfs)>0 else pd.DataFrame(dict(name=[]))
        if len(tmp)>0:
            chemnet1 = tmp.groupby('label',as_index=False).max()[['label','plddt','plddt_lp','lddt']]
            chemnet2 = tmp.groupby('label',as_index=False).min()[['label','lrmsd','kabsch']]
            chemnet3 = tmp.groupby('label',as_index=False).mean()[['label','lrmsd','kabsch']]
            colnames = tmp.columns[1:]
            chemnet = chemnet1.merge(chemnet2, on='label').rename(
                columns={col:'cn_'+col+'_best' for col in colnames})
            chemnet = chemnet.merge(chemnet3, on='label').rename(
                columns={col:'cn_'+col+'_mean' for col in colnames})
            chemnet = chemnet.rename(columns={'label':'name'})
            df_accum = df_accum.merge(chemnet, on='name', how='outer')

        # rosetta ligand
        df_s = [pd.read_csv(fn,index_col=None) for fn in glob.glob(mpnn_dir+'/rosettalig_scores.csv.*')]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        if len(tmp)>0:
            df_accum = df_accum.merge(tmp, on='name', how='outer')

        # mpnn likelihoods
        mpnn_scores = load_mpnn_scores(mpnn_dir)
        df_accum = df_accum.merge(mpnn_scores, on='name', how='outer')
        df_accum['mpnn_index'] = df_accum.name.map(lambda x: int(x.split('_')[-1]))
        df_accum['name'] = df_accum.name.map(lambda x: '_'.join(x.split('_')[:-1]))
        df_out = df_base.copy().merge(df_accum, on='name', how='outer')
        return df_out

    # MPNN metrics
    if os.path.exists(args.datadir+'/mpnn/'):
        df_mpnn = _load_mpnn_df(args.datadir+'/mpnn/', df_base)
        if df_mpnn.shape[1] > df_base.shape[1]: # were there designs that we added metrics for?
            df_mpnn['mpnn'] = True
            df_mpnn['ligmpnn'] = False
            df_all_list.append(df_mpnn)

    # LigandMPNN metrics
    if os.path.exists(args.datadir+'/ligmpnn/'):
        df_ligmpnn = _load_mpnn_df(args.datadir+'/ligmpnn/', df_base)
        if df_ligmpnn.shape[1] > df_base.shape[1]: # were there designs that we added metrics for?
            df_ligmpnn['mpnn'] = False
            df_ligmpnn['ligmpnn'] = True
            df_all_list.append(df_ligmpnn)

    # concatenate all designs into one list
    df = pd.concat(df_all_list)

    # add seq/struc clusters (assumed to be the same for mpnn designs as non-mpnn)
    for path in [
        args.datadir+'/tm_clusters.csv',
        args.datadir+'/blast_clusters.csv',
    ]:
        df_s = [ pd.read_csv(fn,index_col=0) for fn in glob.glob(path) ]
        tmp = pd.concat(df_s) if len(df_s)>0 else pd.DataFrame(dict(name=[]))
        df = df.merge(tmp, on='name', how='outer')

    df.to_csv(args.datadir+'/'+args.outcsv, index=None)
    print(f'Wrote metrics dataframe {df.shape} to "{args.datadir}/{args.outcsv}"')

def load_mpnn_scores(folder):

    filenames = glob.glob(folder+'/seqs/*.fa')

    records = []
    for fn in filenames:
        scores = []
        with open(fn) as f:
            lines = f.readlines()
            for header in lines[2::2]:
                scores.append(float(header.split(',')[2].split('=')[1]))

            for i, score in enumerate(scores):
                records.append(dict(
                    name = os.path.basename(fn).replace('.fa','') + f'_{i}',
                    mpnn_score = score
                ))

    df = pd.DataFrame.from_records(records)
    return df

if __name__ == "__main__":
    main()
