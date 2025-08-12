#!/usr/bin/env python3
import os
import csv
import argparse
import yaml
import glob
import fnmatch

def _norm_sub(s):
    s = s.strip().strip('"').strip("'")
    return s if s.startswith('sub-') else f"sub-{s}"

def _norm_ses(s):
    if not s: return None
    s = str(s).strip()
    if s.startswith('ses-'):
        return s
    try:
        return f"ses-{int(float(s)):02d}"
    except ValueError:
        return f"ses-{s}"

def _iter_unwrap_lines(fp):
    """Yield lines with outer quotes removed if the entire line is quoted."""
    for line in fp:
        line = line.rstrip("\r\n")
        if len(line) >= 2 and line.startswith('"') and line.endswith('"'):
            yield line[1:-1]
        else:
            yield line

def _robust_dict_reader(path):
    with open(path, newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel_tab

        # Peek the header to detect the "one giant quoted field" case
        peek = next(csv.reader([f.readline()], dialect=dialect))
        f.seek(0)
        if len(peek) == 1 and ("," in peek[0] or "\t" in peek[0]):
            # Whole-line quoted → unwrap lines
            reader = csv.DictReader(_iter_unwrap_lines(f), dialect=dialect)
        else:
            reader = csv.DictReader(f, dialect=dialect)

        for row in reader:
            yield { (k.strip() if isinstance(k,str) else k):
                    (v.strip() if isinstance(v,str) else v)
                    for k,v in row.items() }

def load_age_lookup(bids_root, layout, tsv_path=None, pid_col='participant_id',
                    ses_col='session', age_col='age', age_units='years'):
    """
    Returns:
      - long: dict[(sub, ses)] = age_months
      - cross: dict[sub] = age_months
    """
    if tsv_path is None:
        tsv_path = os.path.join(bids_root, 'participants.tsv')
    if not os.path.exists(tsv_path):
        print('The participants.tsv file does not exist...')
        return {}

    # robust delimiter detection (tab/csv)
    with open(tsv_path, newline='') as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel_tab
        reader = csv.DictReader(f, dialect=dialect)
        lut = {}
        for row in _robust_dict_reader(tsv_path):
            pid = row.get(pid_col)
            if not pid:
                continue
            sub = _norm_sub(pid)
            age_raw = row.get(age_col, "")
            if age_raw in ("", "NA", "NaN", None):
                continue
            try:
                age = float(age_raw)
            except ValueError:
                continue
            age_mo = age*12.0 if age_units.lower().startswith("year") else age
            if layout == "long":
                ses = _norm_ses(row.get(ses_col, ""))
                if not ses:
                    continue
                lut[(sub, ses)] = age_mo
            else:
                lut[sub] = age_mo
        print(lut)
        return lut
    
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

def first_sub_from_path(path, root):
    """Return the first component like 'sub-XXXX' from path relative to root."""
    rel = os.path.relpath(path, root).replace(os.sep, '/')
    for part in rel.split('/'):
        if part.startswith('sub-'):
            return part
    return None

def process_dir(root, glob_suffix, layout, args, allowed_subs, age_lut):
    """
    Scan one BIDS root for images, applying:
      - exclude.yaml patterns
      - optional subject filter (allowed_subs)
      - optional age filter using age_lut (in months), with fallback to parsing 'ses-XXmo'
    Returns a set of matching absolute file paths.
    """
    print(f"Processing {root}")

    # --- Helpers (local) -----------------------------------------------------
    def _rel(path):
        return os.path.relpath(path, root).replace(os.sep, '/')

    def _first_sub_from_path(path):
        for part in _rel(path).split('/'):
            if part.startswith('sub-'):
                return part
        return None

    def _first_ses_from_path(path):
        for part in _rel(path).split('/'):
            if part.startswith('ses-'):
                return part
        return None

    def _age_ok_for_dir(sub, ses):
        """Return True if age passes min/max; False to exclude; None means unknown → exclude."""
        # No filtering requested
        if args.min_age_months is None and args.max_age_months is None:
            return True

        # Prefer participants.tsv lookup
        if layout == 'long':
            age_mo = age_lut.get((sub, ses))
        else:
            age_mo = age_lut.get(sub)

        # Fallback: parse ses-XXmo if not in TSV
        if age_mo is None and ses:
            if ses.endswith('wk'):
                return False  # always skip week sessions
            if ses.endswith('mo'):
                try:
                    age_mo = int(ses[len('ses-'):-len('mo')])
                except Exception:
                    age_mo = None

        if age_mo is None:
            return False  # unknown age → exclude conservatively

        if args.min_age_months is not None and age_mo < args.min_age_months:
            return False
        if args.max_age_months is not None and age_mo > args.max_age_months:
            return False
        return True

    # --- Excludes ------------------------------------------------------------
    yaml_file = os.path.join(root, args.exclude_file)
    raw = load_excludes(yaml_file) if os.path.exists(yaml_file) else []
    runs = [e for e in raw if '_run-' in e]
    gens = [e for e in raw if '_run-' not in e]
    patterns = expand_excludes(gens)

    # Add run-level exact patterns
    for run_id in runs:
        session_key, _ = run_id.rsplit('_run-', 1)  # sub-XXX_ses-YYY
        session_path = session_key.replace('_ses-', '/ses-')
        if args.modality in ('T1w', 'T2w'):
            pat = f"{session_path}/anat/{run_id}_{args.modality}.nii.gz"
        else:
            pat = f"**/{run_id}_*.nii.gz"
        patterns.append(pat)

    found = set()

    # --- T1w/T2w (anat) branch ----------------------------------------------
    if args.modality in ('T1w', 'T2w'):
        anat_glob = (os.path.join(root, 'sub-*', 'ses-*', 'anat')
                     if layout == 'long' else
                     os.path.join(root, 'sub-*', 'anat'))

        for anat_dir in glob.glob(anat_glob):
            sub = _first_sub_from_path(anat_dir)
            if allowed_subs and (sub not in allowed_subs):
                continue

            ses = _first_ses_from_path(anat_dir) if layout == 'long' else None
            # age filter at directory level
            if not _age_ok_for_dir(sub, ses):
                continue

            # collect images
            for img in glob.glob(os.path.join(anat_dir, f'*_{args.modality}.nii.gz')):
                if is_excluded(img, patterns, root):
                    continue
                # safety: re-check subject filter
                if allowed_subs:
                    sub_img = _first_sub_from_path(img)
                    if sub_img not in allowed_subs:
                        continue
                found.add(os.path.abspath(img))

    # --- Other modalities (recursive pattern) --------------------------------
    else:
        for img in glob.glob(os.path.join(root, glob_suffix), recursive=True):
            sub = _first_sub_from_path(img)
            if allowed_subs and (sub not in allowed_subs):
                continue

            ses = _first_ses_from_path(img) if layout == 'long' else None
            if not _age_ok_for_dir(sub, ses):
                continue

            if is_excluded(img, patterns, root):
                continue

            found.add(os.path.abspath(img))

    print(f"  found {len(found)} images")
    return found

def norm_sub_id(s):
    s = s.strip().strip('"').strip("'")
    return s if s.startswith('sub-') else f"sub-{s}"

def load_subjects(subjects_list, subjects_file):
    subs = set()
    if subjects_list:
        subs.update(norm_sub_id(x) for x in subjects_list)
    if subjects_file and os.path.exists(subjects_file):
        with open(subjects_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    subs.add(norm_sub_id(line))
    return subs

if __name__ == '__main__':
    # Example command:
    # python create_input_txt.py "/home/andjela/joplin-intra-inter/hc-bcp" -l long --modality T1w -e exclude.yaml -o to_preprocess_bcp_all_except_exclude.txt
    # python create_input_txt.py "/home/andjela/joplin-intra-inter/hc-new-england" -l long \
    #   --modality T1w \
    #   --min-age-months 12 --max-age-months 84 \
    #   --age-tsv /home/andjela/joplin-intra-inter/hc-new-england/participants.tsv --age-col age --age-units months \
    #   -o to_preprocess_new_england.txt

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
    parser.add_argument('--min-age-months', type=int, default=None,
                        help="If set, exclude any session younger than this (months)")
    parser.add_argument('--max-age-months', type=int, default=None,
                    help="If set, exclude any session older than this (months)")
    parser.add_argument('--subjects', nargs='*', default=None,
                        help="Explicit list of subject IDs (sub-XXXX). If set, only these subjects are considered.")
    parser.add_argument('--subjects-file', default=None,
                        help="Text file with one subject ID per line; combined with --subjects if both given.")
    parser.add_argument('--age-tsv', default=None,
                    help="Use age from this TSV (defaults to <bids_root>/participants.tsv)")
    parser.add_argument('--age-pid-col', default='participant_id',
                        help="Column name for participant id in age TSV (default: participant_id)")
    parser.add_argument('--age-ses-col', default='session',
                        help="Column name for session in age TSV (default: session)")
    parser.add_argument('--age-col', default='age',
                        help="Column name for age in age TSV (default: age)")
    parser.add_argument('--age-units', choices=['years','months'], default='years',
                        help="Units of the age column (default: years)")
    args = parser.parse_args()
    if len(args.layouts) != len(args.bids_dirs):
        parser.error("--layouts must match number of bids_dirs")

    allowed_subs = load_subjects(args.subjects, args.subjects_file)

    # Build one age lookup per BIDS root (falls back to <root>/participants.tsv when --age-tsv is None)
    age_lookups = {}
    for root, layout in zip(args.bids_dirs, args.layouts):
        root_abs = os.path.abspath(root)
        age_lookups[root_abs] = load_age_lookup(
            bids_root=root_abs,
            layout=layout,
            tsv_path=args.age_tsv,             # None → default to <root_abs>/participants.tsv inside load_age_lookup
            pid_col=args.age_pid_col,
            ses_col=args.age_ses_col,
            age_col=args.age_col,
            age_units=args.age_units
        )

    # Collect matches
    final = set()
    for root, layout in zip(args.bids_dirs, args.layouts):
        root_abs = os.path.abspath(root)
        if args.pattern:
            glob_s = args.pattern
        elif args.modality not in ('T1w', 'T2w') and args.modality:
            glob_s = (f"sub-*/ses-*/*/*_{args.modality}.nii.gz" if layout == 'long'
                    else f"sub-*/*/*_{args.modality}.nii.gz")
        else:
            glob_s = "**/*.nii.gz"

        final |= process_dir(
            root_abs, glob_s, layout, args,
            allowed_subs=allowed_subs,
            age_lut=age_lookups.get(root_abs, {})
        )

    print(f"Writing {len(final)} paths to {args.output}")
    with open(args.output, 'w') as out:
        for pth in sorted(final):
            out.write(f'"{pth}"\n')

