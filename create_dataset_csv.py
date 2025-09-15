#!/usr/bin/env python3
"""
This script aggregates preprocessed BIDS image lists across one or more datasets
into a single CSV suitable for downstream modeling. For each dataset it:

  1. Reads a per-dataset “preprocess_<dataset>.txt” listing brain NIfTIs to include.
  2. Loads demographic metadata:
       • participants.tsv for sex (M/F → 0/1) and cross-sectional age
         (converted to months if --age-unit=y).
       • sessions.tsv for session-specific age when layout="long".
  3. Builds per-record file paths:
       image_path  = {dest_path}/{<sub>[_<ses>]}_brain.nii.gz  
       segm_path   = {dest_path}/{<sub>[_<ses>]}_segm.nii.gz  
       latent_path = {dest_path}/{<sub>[_<ses>]}_latent.npz  
  4. Groups subjects by sex and age-bin, then assigns each to one of N folds
     using scikit-learn’s StratifiedKFold (default 5).
  5. Reports counts:
       • number of unique subjects per fold
       • if layout="long", also number of scans per fold
  6. Normalizes age:
       • Copies raw age into column `age_bef_norm`
       • Min–max scales age across all records to [0,1] in column `age`
  7. Outputs a combined CSV with one row per subject/scan.

Arguments:
    --bids-roots             List of BIDS root directories (e.g. /data/hc-bcp)
    --layouts                Matching list of "long" or "cross" for each root
    --input-lists            Matching list of preprocess_<dataset>.txt files
    --age-unit               "m" = months, "y" = years (converted to months)
    --dest-path-for-images   Base folder for writing brain/segm/latent files
    --out-csv                Path to output CSV (default: dataset.csv)
    --folds                  Number of stratified folds (default: 5)
    --seed                   Random seed for fold assignment (default: 42)

Outputs:
    A single CSV (--out-csv) with one row per image, containing:
      • dataset        — dataset name (basename of BIDS root)  
      • subject_id     — BIDS subject label (sub-XXX)  
      • image_uid      — subject or session identifier  
      • sex            — 0=M, 1=F, -1=missing  
      • age            — normalized age [0,1]  
      • age_bef_norm   — raw age in months before normalization  
      • image_path     — path to “_brain.nii.gz”  
      • segm_path      — path to “_segm.nii.gz”  
      • latent_path    — path to “_latent.npz”  
      • split          — integer fold assignment (1..N)  

Console output additionally reports per-fold subject/scan counts and min/max
age values before normalization.

Authors:
    Andjela Dimitrijevic
"""  

import pandas as pd
import argparse
import random
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def load_participants(root, age_unit="m"):
    """
    Load participants.tsv and return dicts for sex and age.
    
    Parameters
    ----------
    root : str or Path
        Root directory containing participants.tsv
    age_unit : str, optional
        Unit of age values: 
          - "m" = months (keep as is)
          - "y" = years (convert to months)
    """
    print(f'Root is {root}')
    pfile = Path(root) / 'participants.tsv'
    df = pd.read_csv(pfile, sep='\t')
    df.columns = df.columns.str.strip()

    # ---- Normalize sex/gender column
    sex_col = None
    for candidate in ['sex', 'Sex', 'gender', 'Gender']:
        if candidate in df.columns:
            sex_col = candidate
            break
    
    if sex_col:
        col = df[sex_col]

        # 1) Try to keep existing 0/1 (works for numeric and "0"/"1")
        sex_num = pd.to_numeric(col, errors='coerce').where(lambda x: x.isin([0, 1]))

        # 2) Map remaining values (M/F, case-insensitive; add variants if you like)
        sex_map = col.astype(str).str.strip().str.lower().map({
            'm': 0, 'male': 0,
            'f': 1, 'female': 1
        })

        # 3) Combine (prefer numeric where present)
        df['sex'] = sex_num.fillna(sex_map).astype('Int64')  # keeps NA if unknown
    else:
        df['sex'] = None

    # ---- Normalize age column
    age_col = None
    for candidate in ['age', 'Age']:
        if candidate in df.columns:
            age_col = candidate
            break

    if age_col:
        df['age'] = df[age_col]
        
        if age_unit.lower() == "y":
            df['age'] = df['age'] * 12.0
        # if "m", keep as-is
    else:
        df['age'] = None

    # ---- Build dicts
    sex_map = df.set_index('participant_id')['sex'].to_dict()
    age_map = df.set_index('participant_id')['age'].to_dict()
    

    return sex_map, age_map


# def load_sessions(root, age_unit="m"):
#     sfile = Path(root) / 'sessions.tsv'
#     df = pd.read_csv(sfile, sep='\t')
#     df.columns = df.columns.str.strip()
#     age_col = 'age' if 'age' in df.columns else ('Age' if 'Age' in df.columns else None)
#     if age_col:
#         df['age'] = pd.to_numeric(df[age_col], errors='coerce')
        
#         if age_unit == 'y':
#             df['age'] = df['age'] * 12.0
#     else:
#         df['age'] = None
#     return df

def load_sessions(root, age_unit="m"):
    from pathlib import Path
    import pandas as pd

    sfile = Path(root) / 'sessions.tsv'
    df = pd.read_csv(sfile, sep='\t')
    df.columns = df.columns.str.strip()

    # Pick the source session column
    ses_src = 'session_id' if 'session_id' in df.columns else (
        'session' if 'session' in df.columns else None
    )

    if ses_src is not None:
        def _norm_ses(v):
            if pd.isna(v):
                return pd.NA
            s = str(v).strip()
            # strip leading "ses-" if present
            if s.lower().startswith('ses-'):
                s = s[4:]
            # numeric? -> ses-XX
            try:
                n = int(float(s))
                return f"ses-{n:02d}"
            except ValueError:
                # not numeric; just ensure it has ses- prefix
                return f"ses-{s}"

        df['session_id'] = df[ses_src].apply(_norm_ses)

    # Age column (unchanged from your version)
    age_col = 'age' if 'age' in df.columns else ('Age' if 'Age' in df.columns else None)
    if age_col:
        df['age'] = pd.to_numeric(df[age_col], errors='coerce')
        if age_unit.lower() == 'y':
            df['age'] = df['age'] * 12.0
    else:
        df['age'] = None

    return df

def _normalize_units_list(units_arg, n, parser):
    # default: assume months everywhere if not provided
    if units_arg is None:
        return ['m'] * n
    units = [u.lower() for u in units_arg]
    # if a single value is provided, broadcast to all roots
    if len(units) == 1:
        return units * n
    if len(units) != n:
        parser.error('Number of --age-units must match number of --bids-roots (or provide a single value to use for all).')
    # validate
    bad = [u for u in units if u not in ('m', 'y')]
    if bad:
        parser.error(f'Invalid age unit(s): {bad}. Use "m" or "y".')
    return units


def main():
    parser = argparse.ArgumentParser(
        description='Create combined dataset CSV with stratified folds and normalized age')
    parser.add_argument('--bids-roots', nargs='+', required=True,
                        help='List of BIDS root directories')
    parser.add_argument('--layouts', nargs='+', choices=['long','cross'], required=True,
                        help='Layout per root: long or cross')
    parser.add_argument('--input-lists', nargs='+', required=True,
                        help='List of preprocess_<dataset>.txt files, one per BIDS root')
    parser.add_argument('--age-units', nargs='+', default=None,
                    help='Age units per dataset, matching --bids-roots order. '
                         'Use "y" for years or "m" for months. '
                         'If one value is given, it is applied to all datasets.')
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

    if args.age_units is None:
        args.age_units = ['m'] * len(args.bids_roots)
    elif len(args.age_units) != len(args.bids_roots):
        parser.error('Number of --age-units must match --bids-roots')

    age_units_per_root = _normalize_units_list(args.age_units, n, parser)

    # load metadata per dataset
    sex_maps = {}
    age_maps = {}
    session_dfs = {}

    for root, layout, unit in zip(args.bids_roots, args.layouts, age_units_per_root):
        ds = Path(root).name
        sex_map, age_map = load_participants(root, age_unit=unit)
        sex_maps[ds] = sex_map
        age_maps[ds] = age_map
        if layout == 'long':
            session_dfs[ds] = load_sessions(root, age_unit=unit)

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
    # print(rows)
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

    # --- counts per split ---
    # subjects per split (unique subjects)
    subj_counts = (
        subj_info
        .assign(split=subj_info['subject_id'].map(fold_assign))
        .groupby('split')['subject_id'].nunique()
        .sort_index()
    )

    print("\nSubjects per split:")
    for sp, n in subj_counts.items():
        print(f"  split {int(sp)}: {int(n)} subjects")

    # scans per split (only if long layout: multiple rows per subject)
    if getattr(args, "layout", None) == "long":
        scan_counts = df.groupby('split').size().sort_index()
        print("\nScans per split (long layout):")
        for sp, n in scan_counts.items():
            print(f"  split {int(sp)}: {int(n)} scans")

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
