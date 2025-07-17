#!/usr/bin/env python3
import os
import argparse
import yaml
import glob
import fnmatch

def load_excludes(yaml_path):
    """Load exclude list from a YAML; supports list or dict with 'exclude'."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('exclude', 'exclude_paths', 'exclude_images'):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Could not parse excludes from {yaml_path!r}")

def expand_excludes(excludes):
    """Expand BIDS-style identifiers into glob patterns."""
    patterns = []
    for e in excludes:
        norm = e.strip().replace(os.sep, '/')
        if '/' in norm or '*' in norm:
            patterns.append(norm)
        elif '_ses-' in norm:
            sub, _, ses = norm.partition('_ses-')
            patterns.append(f"{sub}/ses-{ses}/**")
        else:
            patterns.append(f"{norm}/**")
    return patterns

def is_excluded(path, patterns, root):
    rel = os.path.relpath(path, root).replace(os.sep, '/')
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)

def process_dir(root, glob_suffix, layout, args):
    print(f"Processing {root}")
    # load excludes
    yaml_file = os.path.join(root, args.exclude_file)
    raw = load_excludes(yaml_file) if os.path.exists(yaml_file) else []
    # split excludes
    runs = [e for e in raw if '_run-' in e]
    gens = [e for e in raw if '_run-' not in e]
    # build exclude patterns for generic
    patterns = expand_excludes(gens)
        # add run-level excludes to patterns
    for run_id in runs:
        # derive session directory from run_id
        # run_id example: 'sub-002081_ses-13mo_run-001'
        session_key, _ = run_id.rsplit('_run-', 1)
        session_path = session_key.replace('_ses-', '/ses-')
        if args.modality in ('T1w', 'T2w'):
            # exclude the exact run NIfTI under anat
            patterns.append(f"{session_path}/anat/{run_id}_{args.modality}.nii.gz")
        else:
            patterns.append(f"**/{run_id}_*.nii.gz")

    found = set()
    # restricted search for T1w/T2w in anat dirs
    if args.modality in ('T1w', 'T2w'):
        # choose anat folder pattern based on layout
        if layout == 'long':
            anat_glob = os.path.join(root, 'sub-*', 'ses-*', 'anat')
        else:
            anat_glob = os.path.join(root, 'sub-*', 'anat')
        for ad in glob.glob(anat_glob):
            # get session age and skip if <12mo
            parts = ad.replace(os.sep, '/').split('/')
            # find 'ses-XXmo' in parts
            ses_part = next((p for p in parts if p.startswith('ses-') and p.endswith('mo')), None)
            if ses_part:
                try:
                    age = int(ses_part.replace('ses-','').replace('mo',''))
                    if age < 12:
                        continue
                except ValueError:
                    pass
            # now scan images
            for img in glob.glob(os.path.join(ad, f'*_{args.modality}.nii.gz')):
                if is_excluded(img, patterns, root):
                    continue
                found.add(img)
    else:
        # full recursive search, also skip sessions <12mo
        for img in glob.glob(os.path.join(root, glob_suffix), recursive=True):
            # skip sessions <12mo in path
            rel = os.path.relpath(img, root).replace(os.sep, '/')
            parts = rel.split('/')
            ses_part = next((p for p in parts if p.startswith('ses-') and p.endswith('mo')), None)
            if ses_part:
                try:
                    age = int(ses_part.replace('ses-','').replace('mo',''))
                    if age < 12:
                        continue
                except ValueError:
                    pass
            if is_excluded(img, patterns, root):
                continue
            found.add(img)

    print(f"  found {len(found)} images")
    return found

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="List BIDS images with per-dataset layout")
    p.add_argument('bids_dirs', nargs='+', help="BIDS root directories")
    p.add_argument('-l', '--layouts', nargs='+', choices=['long', 'cross'],
                   required=True, help="Layout for each dir: 'long' or 'cross'")
    p.add_argument('-e', '--exclude-file', default='exclude.yaml',
                   help="YAML file listing identifiers to skip")
    p.add_argument('-m', '--modality', choices=['T1w', 'T2w', 'FLAIR', 'bold', 'dwi'],
                   help="Suffix to include, e.g. T1w, dwi")
    p.add_argument('-p', '--pattern', default=None,
                   help="Override glob pattern for non-T1w/T2w")
    p.add_argument('-o', '--output', required=True, help="Output .txt file")
    args = p.parse_args()

    if len(args.layouts) != len(args.bids_dirs):
        p.error("--layouts must match number of BIDS dirs")

    final = set()
    for root, layout in zip(args.bids_dirs, args.layouts):
        root = os.path.abspath(root)
        # determine recursive pattern for non-anat cases
        if args.pattern:
            glob_s = args.pattern
        elif args.modality not in ('T1w', 'T2w') and args.modality:
            glob_s = f"sub-*/ses-*/*/*_{args.modality}.nii.gz" if layout=='long' else f"sub-*/*/*_{args.modality}.nii.gz"
        else:
            glob_s = '**/*.nii.gz'

        final |= process_dir(root, glob_s, layout, args)

    print(f"Writing {len(final)} paths to {args.output}")
    with open(args.output, 'w') as out:
        for pth in sorted(final):
            out.write(pth + '\n')
