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
    # load raw excludes
    yaml_file = os.path.join(root, args.exclude_file)
    raw = load_excludes(yaml_file) if os.path.exists(yaml_file) else []
    # separate run-level and generic excludes
    runs = [e for e in raw if '_run-' in e]
    gens = [e for e in raw if '_run-' not in e]
    patterns = expand_excludes(gens)
    # add run-level exact patterns
    for run_id in runs:
        session_key, _ = run_id.rsplit('_run-', 1)
        session_path = session_key.replace('_ses-', '/ses-')
        pat = f"{session_path}/anat/{run_id}_{args.modality}.nii.gz" if args.modality in ('T1w','T2w') else f"**/{run_id}_*.nii.gz"
        patterns.append(pat)

    found = set()
    # scan anatomicals for T1w/T2w only
    if args.modality in ('T1w', 'T2w'):
        # identify anat dirs by layout
        if layout == 'long':
            anat_glob = os.path.join(root, 'sub-*', 'ses-*', 'anat')
        else:
            anat_glob = os.path.join(root, 'sub-*', 'anat')
        for ad in glob.glob(anat_glob):
            # parse session label
            parts = ad.replace(os.sep, '/').split('/')
            ses_part = next((p for p in parts if p.startswith('ses-') and (p.endswith('mo') or p.endswith('wk'))), None)
            if ses_part:
                # skip all weeks
                if ses_part.endswith('wk'):
                    continue
                # skip months <12
                try:
                    age = int(ses_part[len('ses-'):-len('mo')])
                    if age < 12:
                        continue
                except ValueError:
                    continue
            # scan files
            for img in glob.glob(os.path.join(ad, f'*_{args.modality}.nii.gz')):
                if is_excluded(img, patterns, root):
                    continue
                found.add(img)
    else:
        # full recursive scan for other modalities
        for img in glob.glob(os.path.join(root, glob_suffix), recursive=True):
            # skip any wk sessions
            rel = os.path.relpath(img, root).replace(os.sep,'/')
            parts = rel.split('/')
            ses_part = next((p for p in parts if p.startswith('ses-') and (p.endswith('mo') or p.endswith('wk'))), None)
            if ses_part:
                if ses_part.endswith('wk'):
                    continue
                try:
                    age = int(ses_part[len('ses-'):-len('mo')])
                    if age < 12:
                        continue
                except ValueError:
                    continue
            if is_excluded(img, patterns, root):
                continue
            found.add(img)

    print(f"  found {len(found)} images")
    return found

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="List BIDS images with per-dataset layout")
    parser.add_argument('bids_dirs', nargs='+', help="BIDS root directories")
    parser.add_argument('-l', '--layouts', nargs='+', choices=['long','cross'], required=True,
                        help="Layout per dir: 'long' or 'cross'")
    parser.add_argument('-e', '--exclude-file', default='exclude.yaml',
                        help="YAML file listing identifiers to skip")
    parser.add_argument('-m', '--modality', choices=['T1w','T2w','FLAIR','bold','dwi'],
                        help="Suffix to include, e.g. T1w, dwi")
    parser.add_argument('-p', '--pattern', default=None,
                        help="Override glob pattern for non-anat scans")
    parser.add_argument('-o', '--output', required=True, help="Output .txt file")
    args = parser.parse_args()
    if len(args.layouts) != len(args.bids_dirs):
        parser.error("--layouts must match number of bids_dirs")

    # choose fallback glob
    final = set()
    for root, layout in zip(args.bids_dirs, args.layouts):
        root = os.path.abspath(root)
        if args.pattern:
            glob_s = args.pattern
        elif args.modality not in ('T1w','T2w') and args.modality:
            glob_s = f"sub-*/ses-*/*/*_{args.modality}.nii.gz" if layout=='long' else f"sub-*/*/*_{args.modality}.nii.gz"
        else:
            glob_s = '**/*.nii.gz'
        final |= process_dir(root, glob_s, layout, args)

    print(f"Writing {len(final)} paths to {args.output}")
    with open(args.output, 'w') as out:
        for pth in sorted(final): out.write(pth+'\n')
