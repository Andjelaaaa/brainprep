#!/usr/bin/env python3
"""
create_input_txt
================

Create an input list of BIDS images to preprocess.

This script scans one or more BIDS dataset roots and writes a text file
containing one **quoted absolute path** per line (e.g., T1w images). It supports:

- Longitudinal (``sub-*/ses-*/anat``) and cross-sectional (``sub-*/anat``) layouts
- Excluding subjects/sessions/runs via ``exclude.yaml``
- Optional filtering by a list of subjects
- Optional filtering by age range (in months), using ``participants.tsv`` or a custom TSV

By default, the exclusion file is expected at::

    <bids_root>/code/qc/raw/exclude.yaml

The output text file can be passed directly to your preprocessing script
(e.g., ``brainprep.py --inputs <output.txt>``).

Notes for Sphinx
----------------
- All logic is import-safe (no work performed at import).
- Use :func:`main` as the CLI entrypoint.

Examples
--------
Single longitudinal dataset::

    python create_input_txt.py /path/to/hc-calgary-preschool \
      -l long \
      --modality T1w \
      -o to_preprocess_hc-calgary-preschool.txt

Multiple datasets at once::

    python create_input_txt.py /path/to/hc-bcp /path/to/hc-calgary-preschool \
      -l long long \
      --modality T1w \
      -o to_preprocess_all.txt

With an age filter (1–7 years = 12–84 months)::

    python create_input_txt.py /path/to/hc-calgary-preschool \
      -l long \
      --modality T1w \
      --min-age-months 12 --max-age-months 84 \
      --age-tsv /path/to/hc-calgary-preschool/participants.tsv \
      --age-col age --age-units years \
      -o to_preprocess_12to84mo.txt
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import yaml


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _norm_sub(s: str) -> str:
    """Normalize a participant label into a BIDS-like ``sub-...`` form."""
    s = str(s).strip().strip('"').strip("'")
    return s if s.startswith("sub-") else f"sub-{s}"


def _norm_ses(s: Union[str, int, float, None]) -> Optional[str]:
    """Normalize a session label into ``ses-XX`` (numeric) or ``ses-<str>``."""
    if s is None or str(s).strip() == "":
        return None
    s = str(s).strip()
    if s.startswith("ses-"):
        return s
    try:
        return f"ses-{int(float(s)):02d}"
    except ValueError:
        return f"ses-{s}"


def _iter_unwrap_lines(fp: Iterable[str]) -> Iterator[str]:
    """Yield lines with outer quotes removed if the entire line is quoted."""
    for line in fp:
        line = line.rstrip("\r\n")
        if len(line) >= 2 and line.startswith('"') and line.endswith('"'):
            yield line[1:-1]
        else:
            yield line


def _robust_dict_reader(path: str) -> Iterator[Dict[str, str]]:
    """
    CSV/TSV DictReader that handles:
    - auto delimiter sniffing
    - "whole line quoted" files (a single field containing embedded separators)
    """
    with open(path, newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel_tab

        # Peek header to detect "one giant quoted field" case
        peek = next(csv.reader([f.readline()], dialect=dialect))
        f.seek(0)

        if len(peek) == 1 and ("," in peek[0] or "\t" in peek[0]):
            reader = csv.DictReader(_iter_unwrap_lines(f), dialect=dialect)
        else:
            reader = csv.DictReader(f, dialect=dialect)

        for row in reader:
            yield {
                (k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v)
                for k, v in row.items()
            }


# -----------------------------------------------------------------------------
# Age lookup
# -----------------------------------------------------------------------------
def load_age_lookup(
    bids_root: str,
    layout: str,
    tsv_path: Optional[str] = None,
    pid_col: str = "participant_id",
    ses_col: str = "session",
    age_col: str = "age",
    age_units: str = "years",
) -> Dict[Union[str, Tuple[str, str]], float]:
    """
    Load ages from a TSV/CSV into a lookup dictionary.

    For ``layout="long"`` this returns::

        {(sub_id, ses_id): age_months, ...}

    For ``layout="cross"`` this returns::

        {sub_id: age_months, ...}

    Parameters
    ----------
    bids_root:
        BIDS dataset root.
    layout:
        ``"long"`` or ``"cross"``.
    tsv_path:
        Optional TSV/CSV path. If None, defaults to ``<bids_root>/participants.tsv``.
    pid_col, ses_col, age_col:
        Column names in the TSV.
    age_units:
        Either ``"years"`` or ``"months"``. Years are converted to months.

    Returns
    -------
    dict
        Age lookup in months.
    """
    if tsv_path is None:
        tsv_path = os.path.join(bids_root, "participants.tsv")

    if not os.path.exists(tsv_path):
        print("The participants.tsv file does not exist...")
        return {}

    lut: Dict[Union[str, Tuple[str, str]], float] = {}

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

        age_mo = age * 12.0 if age_units.lower().startswith("year") else age

        if layout == "long":
            ses = _norm_ses(row.get(ses_col, ""))
            if not ses:
                continue
            lut[(sub, ses)] = age_mo
        else:
            lut[sub] = age_mo

    return lut


# -----------------------------------------------------------------------------
# Excludes handling
# -----------------------------------------------------------------------------
def load_excludes(yaml_path: str) -> List[str]:
    """
    Load exclude identifiers from YAML.

    Supports either:
    - a YAML list
    - a dict containing a list under keys: ``exclude``, ``exclude_paths``, ``exclude_images``
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("exclude", "exclude_paths", "exclude_images"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Could not parse excludes from {yaml_path!r}")


def expand_excludes(excludes: Sequence[str]) -> List[str]:
    """
    Expand BIDS-style identifiers into glob-like relative patterns.

    Examples
    --------
    - ``sub-10001`` -> ``sub-10001/**``
    - ``sub-10001_ses-001`` -> ``sub-10001/ses-001/**``
    - path-like patterns are kept as-is.
    """
    patterns: List[str] = []
    for e in excludes:
        norm = e.strip().replace(os.sep, "/")
        if "/" in norm or "*" in norm:
            patterns.append(norm)
        elif "_ses-" in norm:
            sub, _, ses = norm.partition("_ses-")
            patterns.append(f"{sub}/ses-{ses}/**")
        else:
            patterns.append(f"{norm}/**")
    return patterns


def is_excluded(path: str, patterns: Sequence[str], root: str) -> bool:
    """Check whether an absolute path is excluded by any relative pattern."""
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


# -----------------------------------------------------------------------------
# Core scanning logic
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ScanArgs:
    """Arguments used during scanning (subset of CLI args)."""

    exclude_file: str
    modality: Optional[str]
    min_age_months: Optional[int]
    max_age_months: Optional[int]
    pattern: Optional[str]


def process_dir(
    root: str,
    glob_suffix: str,
    layout: str,
    scan_args: ScanArgs,
    allowed_subs: Set[str],
    age_lut: Dict[Union[str, Tuple[str, str]], float],
) -> Set[str]:
    """
    Scan one BIDS dataset root and return a set of matching absolute file paths.

    Applies:
    - exclusion patterns from ``exclude.yaml``
    - optional subject filter
    - optional age filter (months), with fallback to parsing ``ses-XXmo`` if present

    Parameters
    ----------
    root:
        Absolute dataset root.
    glob_suffix:
        Glob pattern to use for non-anat modalities (or fallback scanning).
    layout:
        ``"long"`` or ``"cross"``.
    scan_args:
        Scanning parameters (exclude file, modality, age range, pattern).
    allowed_subs:
        Set of allowed subject IDs (e.g. ``{"sub-10001", ...}``). Empty means no restriction.
    age_lut:
        Age lookup in months.

    Returns
    -------
    set[str]
        Matching image paths (absolute).
    """
    print(f"Processing {root}")

    def _rel(path: str) -> str:
        return os.path.relpath(path, root).replace(os.sep, "/")

    def _first_sub_from_path(path: str) -> Optional[str]:
        for part in _rel(path).split("/"):
            if part.startswith("sub-"):
                return part
        return None

    def _first_ses_from_path(path: str) -> Optional[str]:
        for part in _rel(path).split("/"):
            if part.startswith("ses-"):
                return part
        return None

    def _age_ok_for_dir(sub: str, ses: Optional[str]) -> bool:
        # No filtering requested
        if scan_args.min_age_months is None and scan_args.max_age_months is None:
            return True

        # Prefer TSV lookup
        if layout == "long":
            age_mo = age_lut.get((sub, ses)) if ses else None
        else:
            age_mo = age_lut.get(sub)

        # Fallback: parse ses-XXmo if not in TSV
        if age_mo is None and ses:
            if ses.endswith("wk"):
                return False
            if ses.endswith("mo"):
                try:
                    age_mo = int(ses[len("ses-") : -len("mo")])
                except Exception:
                    age_mo = None

        if age_mo is None:
            return False

        if scan_args.min_age_months is not None and age_mo < scan_args.min_age_months:
            return False
        if scan_args.max_age_months is not None and age_mo > scan_args.max_age_months:
            return False
        return True

    # --- Excludes ------------------------------------------------------------
    yaml_file = os.path.join(root, "code", "qc", "raw", scan_args.exclude_file)
    raw = load_excludes(yaml_file) if os.path.exists(yaml_file) else []

    runs = [e for e in raw if "_run-" in e]
    gens = [e for e in raw if "_run-" not in e]
    patterns = expand_excludes(gens)

    # Add run-level exact patterns
    for run_id in runs:
        session_key, _ = run_id.rsplit("_run-", 1)
        session_path = session_key.replace("_ses-", "/ses-")

        if scan_args.modality in ("T1w", "T2w"):
            pat = f"{session_path}/anat/{run_id}_{scan_args.modality}.nii.gz"
        else:
            pat = f"**/{run_id}_*.nii.gz"
        patterns.append(pat)

    found: Set[str] = set()

    # --- T1w/T2w (anat) branch ----------------------------------------------
    if scan_args.modality in ("T1w", "T2w"):
        anat_glob = (
            os.path.join(root, "sub-*", "ses-*", "anat")
            if layout == "long"
            else os.path.join(root, "sub-*", "anat")
        )

        for anat_dir in glob.glob(anat_glob):
            sub = _first_sub_from_path(anat_dir)
            if sub is None:
                continue
            if allowed_subs and (sub not in allowed_subs):
                continue

            ses = _first_ses_from_path(anat_dir) if layout == "long" else None
            if not _age_ok_for_dir(sub, ses):
                continue

            for img in glob.glob(os.path.join(anat_dir, f"*_{scan_args.modality}.nii.gz")):
                if is_excluded(img, patterns, root):
                    continue
                if allowed_subs:
                    sub_img = _first_sub_from_path(img)
                    if sub_img not in allowed_subs:
                        continue
                found.add(os.path.abspath(img))

    # --- Other modalities (recursive pattern) --------------------------------
    else:
        for img in glob.glob(os.path.join(root, glob_suffix), recursive=True):
            sub = _first_sub_from_path(img)
            if sub is None:
                continue
            if allowed_subs and (sub not in allowed_subs):
                continue

            ses = _first_ses_from_path(img) if layout == "long" else None
            if not _age_ok_for_dir(sub, ses):
                continue

            if is_excluded(img, patterns, root):
                continue

            found.add(os.path.abspath(img))

    print(f"  found {len(found)} images")
    return found


# -----------------------------------------------------------------------------
# Subject filter
# -----------------------------------------------------------------------------
def norm_sub_id(s: str) -> str:
    """Normalize subject IDs provided by the user into ``sub-...`` format."""
    s = str(s).strip().strip('"').strip("'")
    return s if s.startswith("sub-") else f"sub-{s}"


def load_subjects(subjects_list: Optional[Sequence[str]], subjects_file: Optional[str]) -> Set[str]:
    """
    Load allowed subjects from CLI arguments.

    Parameters
    ----------
    subjects_list:
        Subject IDs provided directly on the command line.
    subjects_file:
        Optional file containing one subject ID per line.

    Returns
    -------
    set[str]
        Allowed subject IDs.
    """
    subs: Set[str] = set()
    if subjects_list:
        subs.update(norm_sub_id(x) for x in subjects_list)
    if subjects_file and os.path.exists(subjects_file):
        with open(subjects_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    subs.add(norm_sub_id(line))
    return subs


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="List BIDS images with per-dataset layout")

    parser.add_argument("bids_dirs", nargs="+", help="BIDS root directories")
    parser.add_argument(
        "-l",
        "--layouts",
        nargs="+",
        choices=["long", "cross"],
        required=True,
        help="Layout per dir: 'long' or 'cross'",
    )
    parser.add_argument(
        "-e",
        "--exclude-file",
        default="exclude.yaml",
        help="YAML file listing identifiers to skip (relative to code/qc/raw/)",
    )
    parser.add_argument(
        "-m",
        "--modality",
        choices=["T1w", "T2w", "FLAIR", "bold", "dwi"],
        required=True,
        help="Suffix to include (e.g. T1w, dwi)",
    )
    parser.add_argument("-p", "--pattern", default=None, help="Override glob pattern for non-anat scans")
    parser.add_argument("-o", "--output", required=True, help="Output .txt file")

    parser.add_argument("--min-age-months", type=int, default=None,
                        help="Exclude any session younger than this (months)")
    parser.add_argument("--max-age-months", type=int, default=None,
                        help="Exclude any session older than this (months)")

    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Explicit list of subject IDs (sub-XXXX). If set, only these subjects are considered.")
    parser.add_argument("--subjects-file", default=None,
                        help="Text file with one subject ID per line; combined with --subjects if both given.")

    parser.add_argument("--age-tsv", default=None,
                        help="Use age from this TSV/CSV (default: <bids_root>/participants.tsv)")
    parser.add_argument("--age-pid-col", default="participant_id",
                        help="Column name for participant id in age TSV (default: participant_id)")
    parser.add_argument("--age-ses-col", default="session",
                        help="Column name for session in age TSV (default: session)")
    parser.add_argument("--age-col", default="age",
                        help="Column name for age in age TSV (default: age)")
    parser.add_argument("--age-units", choices=["years", "months"], default="years",
                        help="Units of the age column (default: years)")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entrypoint.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = build_argparser()
    args = parser.parse_args(argv)

    if len(args.layouts) != len(args.bids_dirs):
        parser.error("--layouts must match number of bids_dirs")

    allowed_subs = load_subjects(args.subjects, args.subjects_file)

    scan_args = ScanArgs(
        exclude_file=args.exclude_file,
        modality=args.modality,
        min_age_months=args.min_age_months,
        max_age_months=args.max_age_months,
        pattern=args.pattern,
    )

    # Build one age lookup per root
    age_lookups: Dict[str, Dict[Union[str, Tuple[str, str]], float]] = {}
    for root, layout in zip(args.bids_dirs, args.layouts):
        root_abs = os.path.abspath(root)
        age_lookups[root_abs] = load_age_lookup(
            bids_root=root_abs,
            layout=layout,
            tsv_path=args.age_tsv,
            pid_col=args.age_pid_col,
            ses_col=args.age_ses_col,
            age_col=args.age_col,
            age_units=args.age_units,
        )

    # Collect matches
    final: Set[str] = set()
    for root, layout in zip(args.bids_dirs, args.layouts):
        root_abs = os.path.abspath(root)

        if args.pattern:
            glob_s = args.pattern
        elif args.modality not in ("T1w", "T2w") and args.modality:
            glob_s = (
                f"sub-*/ses-*/*/*_{args.modality}.nii.gz" if layout == "long"
                else f"sub-*/*/*_{args.modality}.nii.gz"
            )
        else:
            glob_s = "**/*.nii.gz"

        final |= process_dir(
            root_abs,
            glob_s,
            layout,
            scan_args,
            allowed_subs=allowed_subs,
            age_lut=age_lookups.get(root_abs, {}),
        )

    print(f"Writing {len(final)} paths to {args.output}")
    with open(args.output, "w") as out:
        for pth in sorted(final):
            out.write(f'"{pth}"\n')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
