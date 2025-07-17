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
    runs = [e for e in raw if '_run-' in e]
    gens = [e for e in raw if '_run-' not in e]
    patterns = expand_excludes(gens)

    found = set()
    # restrict T1w/T2w to anat dirs based on layout
    if args.modality in ('T1w', 'T2w'):
        if layout == 'long':
            anat_glob = os.path.join(root, 'sub-*', 'ses-*', 'anat')
        else:
            anat_glob = os.path.join(root, 'sub-*', 'anat')
        anat_dirs = glob.glob(anat_glob)
        for ad in anat_dirs:
            for img in glob.glob(os.path.join(ad, f'*_{args.modality}.nii.gz')):
                if patterns and is_excluded(img, patterns, root):
                    continue
                found.add(img)
    else:
        # full recursive search
        for img in glob.glob(os.path.join(root, glob_suffix), recursive=True):
            if patterns and is_excluded(img, patterns, root):
                continue
            found.add(img)

    # re-add explicit runs
    for run_id in runs:
        if args.modality in ('T1w', 'T2w'):
            if layout == 'long':
                pat = os.path.join(root, 'sub-*', 'ses-*', 'anat', f"{run_id}_{args.modality}.nii.gz")
            else:
                pat = os.path.join(root, 'sub-*', 'anat', f"{run_id}_{args.modality}.nii.gz")
            matches = glob.glob(pat)
        else:
            pat = os.path.join(root, f"**/{run_id}_{args.modality}.nii.gz") if args.modality else os.path.join(root, f"**/{run_id}_*.nii.gz")
            matches = glob.glob(pat, recursive=True)
        for img in matches:
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
