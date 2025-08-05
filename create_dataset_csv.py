#!/usr/bin/env python3
"""
This script aggregates preprocessed BIDS image lists across one or more datasets
into a single CSV suitable for downstream modeling.  For each dataset it:

  1. Reads a per‐dataset “preprocess_<dataset>.txt” listing brain NIfTIs to include.
  2. Loads demographic metadata:
     - Always reads participants.tsv for sex and (cross‐sectional) age in months.
     - For longitudinal datasets, also reads sessions.tsv for session‐specific age.
  4. Builds file paths for each subject/session:
       image_path = {dest_path}/{<sub>[_<ses>]}_brain.nii.gz  
       segm_path  = {dest_path}/{<sub>[_<ses>]}_segm.nii.gz  
       latent_path= {dest_path}/{<sub>[_<ses>]}_latent.npz  
  5. Stratifies subjects by sex and age‐bin, then assigns each to one of N folds
     (default 5) using scikit‑learn’s StratifiedKFold.
  6. Min–max normalizes the age column across all records into `age` and create `age_bef_norm` column.
  7. Outputs one CSV with columns:
       [dataset, subject_id, session_id, image_uid, sex, age, image_path,
        segm_path, latent_path, fold, age_bef__norm]

Arguments:
    --bids-roots               List of BIDS root directories (e.g. /data/hc-bcp)
    --layouts                  Matching list of “long” or “cross” for each root
    --input-lists              Matching list of preprocess_<dataset>.txt files
    --dest-path-for-images     Base folder for writing brain/segm/latent files
    --out-csv                  Where to write the combined CSV (default: dataset.csv)
    --folds                    Number of stratified folds (default: 5)
    --seed                     Random seed for reproducible splitting (default: 42)

Outputs:
    A single CSV (`--out-csv`) containing one row per image, with:
      • dataset      — dataset name (basename of BIDS root)  
      • subject_id   — BIDS subject label (sub-XXX)  
      • session_id   — BIDS session label (ses-YYY) or blank for cross-sectional  
      • image_uid    — used as a unique image identifier (subject or session)  
      • sex          — 0=M or 1=F  
      • age          — age in months  
      • image_path   — path to “_brain.nii.gz” file under dest path  
      • segm_path    — path to “_segm.nii.gz” file under dest path  
      • latent_path  — path to “_latent.npz” file under dest path  
      • fold         — integer fold assignment (1..N) stratified by sex+age  
      • age_norm     — min–max normalized age in [0,1]

Authors:
    Andjela Dimitrijevic  
"""
import pandas as pd
import argparse
import random
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def load_participants(root):
    pfile = Path(root) / 'participants.tsv'
    df = pd.read_csv(pfile, sep='\t')
    df.columns = df.columns.str.strip()
    df['sex'] = df['sex'].map({'M':0, 'F':1})
    if 'age' in df.columns:
        df['age'] = df['age']
    else:
        df['age'] = None
    sex_map = df.set_index('participant_id')['sex'].to_dict()
    age_map = df.set_index('participant_id')['age'].to_dict()
    return sex_map, age_map


def load_sessions(root):
    sfile = Path(root) / 'sessions.tsv'
    df = pd.read_csv(sfile, sep='\t')
    df.columns = df.columns.str.strip()
    df['age'] = df['age']
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Create combined dataset CSV with stratified folds and normalized age')
    parser.add_argument('--bids-roots', nargs='+', required=True,
                        help='List of BIDS root directories')
    parser.add_argument('--layouts', nargs='+', choices=['long','cross'], required=True,
                        help='Layout per root: long or cross')
    parser.add_argument('--input-lists', nargs='+', required=True,
                        help='List of preprocess_<dataset>.txt files, one per BIDS root')
    parser.add_argument('--dest-path-for-images', required=True,
                        help='Base destination path for brain/segm/latent files')
    parser.add_argument('--out-csv', default='dataset.csv',
                        help='Output CSV file path')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of stratified folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    n = len(args.bids_roots)
    if not (len(args.layouts) == n == len(args.input_lists)):
        parser.error('Number of --bids-roots, --layouts, and --input-lists must match')

    # load metadata per dataset
    sex_maps = {}
    age_maps = {}
    session_dfs = {}
    for root, layout in zip(args.bids_roots, args.layouts):
        ds = Path(root).name
        sex_map, age_map = load_participants(root)
        sex_maps[ds] = sex_map
        age_maps[ds] = age_map
        if layout == 'long':
            session_dfs[ds] = load_sessions(root)

    # gather records
    rows = []
    for root, layout, input_list in zip(args.bids_roots, args.layouts, args.input_lists):
        ds = Path(root).name
        sex_map = sex_maps[ds]
        age_map = age_maps[ds]
        sess_df = session_dfs.get(ds)
        with open(input_list) as f:
            paths = [p.strip() for p in f if p.strip()]
        for p in paths:
            parts = Path(p).parts
            sid = next((x for x in parts if x.startswith('sub-')), None)
            ses = next((x for x in parts if x.startswith('ses-')), None)
            if not sid:
                continue
            image_uid = sid if layout == 'cross' else ses or sid
            # age
            if layout == 'cross':
                age = age_map.get(sid)
            else:
                age = None
                if sess_df is not None and ses:
                    row = sess_df[(sess_df['participant_id']==sid)&(sess_df['session_id']==ses)]
                    if not row.empty:
                        age = float(row.iloc[0]['age'])
            sex = sex_map.get(sid, -1)
            # file base for outputs
            base = sid if layout == 'cross' else f"{sid}_{ses}"
            dest = Path(args.dest_path_for_images)
            rows.append({
                'dataset': ds,
                'subject_id': sid,
                'image_uid': image_uid,
                'sex': sex,
                'age': age,
                'image_path': str(dest / f"{base}_brain.nii.gz"),
                'segm_path': str(dest / f"{base}_segm.nii.gz"),
                'latent_path': str(dest / f"{base}_latent.npz"),
            })

    df = pd.DataFrame(rows)
    # subject-level strata
    subj_info = df.groupby('subject_id').agg({'sex':'first','age':'mean'}).reset_index()
    subj_info['age_bin'] = pd.qcut(subj_info['age'].fillna(subj_info['age'].median()),
                                    q=min(args.folds,5), duplicates='drop').astype(str)
    subj_info['strata'] = subj_info['sex'].astype(str) + '_' + subj_info['age_bin']
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_assign = {}
    for fold, (_, test_idx) in enumerate(skf.split(subj_info, subj_info['strata']), 1):
        for idx in test_idx:
            fold_assign[subj_info.iloc[idx]['subject_id']] = fold
    df['split'] = df['subject_id'].map(fold_assign)

    # rename original age to age_bef_norm
    df['age_bef_norm'] = df['age']
    # normalize age and overwrite 'age'
    ages = df['age_bef_norm']
    min_a, max_a = ages.min(), ages.max()
    df['age'] = ((ages - min_a) / (max_a - min_a)).round(4)
    print(f'min age: {min_a}')
    print(f'max age: {max_a}')
    # drop any leftover age_norm column
    df.drop(columns=['age_norm'], errors='ignore', inplace=True)

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} records to {args.out_csv}")

if __name__ == '__main__':
    main()
