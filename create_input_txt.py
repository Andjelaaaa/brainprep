#!/usr/bin/env python3
import os
import argparse
import yaml
import glob
import fnmatch

def load_excludes(yaml_path):
    """Load exclude list from a YAML. Accepts either:
       - a top-level list, or
       - a dict with a key 'exclude' (or 'exclude_paths' / 'exclude_images')."""
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
    """
    Expand simple BIDS-style identifiers into glob patterns:
      - 'sub-10037_ses-001' -> 'sub-10037/ses-001/**'
      - 'sub-10037'          -> 'sub-10037/**'
      - otherwise pass through (including any explicit paths or globs)
    """
    patterns = []
    for e in excludes:
        e_norm = e.strip().replace(os.sep, '/')
        if '/' in e_norm or '*' in e_norm:
            patterns.append(e_norm)
        elif '_ses-' in e_norm:
            sub, _, rest = e_norm.partition('_ses-')
            patterns.append(f"{sub}/ses-{rest}/**")
        else:
            patterns.append(f"{e_norm}/**")
    return patterns

def is_excluded(path, exclude_patterns, root):
    """Check if `path` (absolute) matches any pattern in `exclude_patterns` (relative to root)."""
    rel = os.path.relpath(path, root).replace(os.sep, '/')
    return any(fnmatch.fnmatch(rel, pat) for pat in exclude_patterns)

def main():
    parser = argparse.ArgumentParser(
        description="List all images under BIDS dirs, excluding those in exclude.yaml"
    )
    parser.add_argument('bids_dirs', nargs='+', help="One or more BIDS root directories")
    parser.add_argument('-e', '--exclude-file', default='exclude.yaml',
                        help="YAML file in each BIDS dir listing identifiers to skip")
    parser.add_argument('-m', '--modality', choices=['T1w', 'T2w', 'FLAIR', 'bold', 'dwi'],
                        help="Modality suffix, e.g. T1w, T2w, bold, dwi")
    parser.add_argument('-p', '--pattern', default=None,
                        help="Custom glob pattern for images (relative to BIDS dir)")
    parser.add_argument('-o', '--output', required=True,
                        help="Output .txt file path (one image per line)")
    args = parser.parse_args()

    # Choose glob pattern for discovery
    if args.modality:
        glob_suffix = f"**/*_{args.modality}.nii.gz"
    elif args.pattern:
        glob_suffix = args.pattern
    else:
        glob_suffix = '**/*.nii.gz'

    final_paths = set()
    for bids_root in args.bids_dirs:
        bids_root = os.path.abspath(bids_root)
        print(f"Processing {bids_root}")

        # Load and split exclude identifiers
        excl_path = os.path.join(bids_root, args.exclude_file)
        raw = load_excludes(excl_path) if os.path.exists(excl_path) else []
        runs = [e.strip() for e in raw if '_run-' in e]
        generics = [e for e in raw if '_run-' not in e]
        patterns = expand_excludes(generics)

        ds_paths = set()
        # discover all matching images
        for img in glob.glob(os.path.join(bids_root, glob_suffix), recursive=True):
            if patterns and is_excluded(img, patterns, bids_root):
                continue
            ds_paths.add(img)

        # explicitly include listed runs
        for run_id in runs:
            if args.modality:
                pat = f"**/{run_id}_{args.modality}.nii.gz"
            else:
                pat = f"**/{run_id}_*.nii.gz"
            for img in glob.glob(os.path.join(bids_root, pat), recursive=True):
                ds_paths.add(img)

        print(f"  found {len(ds_paths)} images")
        final_paths.update(ds_paths)

    print(f"Writing {len(final_paths)} paths to {args.output}")
    with open(args.output, 'w') as out:
        for pth in sorted(final_paths):
            out.write(pth + '\n')

if __name__ == '__main__':
    main()
