#!/usr/bin/env python3
"""
create_dataset_csv
==================
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

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedKFold


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _strip_quotes(s: str) -> str:
    """Strip whitespace and enclosing single/double quotes."""
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1]
    return s


def read_paths_list(txt_path: Path) -> List[str]:
    """
    Read one path per line from a text file.

    Supports lines optionally wrapped in quotes (common when generating lists for tools).

    Parameters
    ----------
    txt_path:
        Path to the text file.

    Returns
    -------
    list[str]
        Cleaned paths.
    """
    out: List[str] = []
    with txt_path.open("r") as f:
        for line in f:
            p = _strip_quotes(line)
            if p:
                out.append(p)
    return out


def detect_sex_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching sex/gender column name, if present."""
    for candidate in ("sex", "Sex", "gender", "Gender"):
        if candidate in df.columns:
            return candidate
    return None


def detect_age_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching age column name, if present."""
    for candidate in ("age", "Age"):
        if candidate in df.columns:
            return candidate
    return None


def normalize_age_to_months(age_series: pd.Series, age_unit: str) -> pd.Series:
    """
    Convert age to months if needed.

    Parameters
    ----------
    age_series:
        Numeric age series.
    age_unit:
        ``"m"`` for months or ``"y"`` for years.

    Returns
    -------
    pd.Series
        Age in months.
    """
    age = pd.to_numeric(age_series, errors="coerce")
    if age_unit.lower() == "y":
        return age * 12.0
    return age


# -----------------------------------------------------------------------------
# Metadata loading
# -----------------------------------------------------------------------------
def load_participants(root: Path, age_unit: str = "m") -> Tuple[Dict[str, int], Dict[str, float]]:
    """
    Load ``participants.tsv`` and return lookup dicts for sex and age.

    Sex is normalized to:
      - 0 = male
      - 1 = female
      - -1 = missing/unknown

    Age is returned in **months**.

    Parameters
    ----------
    root:
        BIDS dataset root containing ``participants.tsv``.
    age_unit:
        ``"m"`` = months, ``"y"`` = years (converted to months).

    Returns
    -------
    sex_map:
        dict mapping ``participant_id`` -> {0,1,-1}
    age_map:
        dict mapping ``participant_id`` -> age_months
    """
    pfile = root / "participants.tsv"
    if not pfile.exists():
        raise FileNotFoundError(f"Missing participants.tsv: {pfile}")

    df = pd.read_csv(pfile, sep="\t")
    df.columns = df.columns.str.strip()

    # ---- Sex
    sex_col = detect_sex_column(df)
    if sex_col is None:
        df["sex_norm"] = -1
    else:
        col = df[sex_col].astype(str).str.strip()

        # numeric 0/1 if present
        sex_num = pd.to_numeric(col, errors="coerce")
        sex_num = sex_num.where(sex_num.isin([0, 1]))

        # string map
        sex_map_str = col.str.lower().map({
            "m": 0, "male": 0,
            "f": 1, "female": 1
        })

        # combine and fill missing as -1
        sex_combined = sex_num.fillna(sex_map_str).fillna(-1).astype(int)
        df["sex_norm"] = sex_combined

    # ---- Age
    age_col = detect_age_column(df)
    if age_col is None:
        df["age_months"] = pd.NA
    else:
        df["age_months"] = normalize_age_to_months(df[age_col], age_unit=age_unit)

    # ---- Dicts keyed by participant_id
    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv must contain a 'participant_id' column")

    sex_map = df.set_index("participant_id")["sex_norm"].to_dict()
    age_map = df.set_index("participant_id")["age_months"].to_dict()
    return sex_map, age_map


def load_sessions(root: Path, age_unit: str = "m") -> pd.DataFrame:
    """
    Load ``sessions.tsv`` and normalize session_id + age.

    For longitudinal datasets, age is looked up using
    ``(participant_id, session_id)``.

    Parameters
    ----------
    root:
        BIDS dataset root containing ``sessions.tsv``.
    age_unit:
        ``"m"`` months, ``"y"`` years (converted to months).

    Returns
    -------
    pd.DataFrame
        Sessions dataframe with columns:
        ``participant_id``, ``session_id``, ``age_months`` (if age column exists).
    """
    sfile = root / "sourcedata/sessions.tsv"
    if not sfile.exists():
        raise FileNotFoundError(f"Missing sessions.tsv: {sfile}")

    df = pd.read_csv(sfile, sep="\t")
    df.columns = df.columns.str.strip()

    if "participant_id" not in df.columns:
        raise ValueError("sessions.tsv must contain a 'participant_id' column")

    # Choose session source column
    ses_src = "session_id" if "session_id" in df.columns else ("session" if "session" in df.columns else None)
    if ses_src is None:
        raise ValueError("sessions.tsv must contain 'session_id' or 'session' column")

    def _norm_ses(v: object) -> str:
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if s.lower().startswith("ses-"):
            s = s[4:]
        # numeric? -> ses-XX
        try:
            n = int(float(s))
            return f"ses-{n:02d}"
        except ValueError:
            return f"ses-{s}"

    df["session_id"] = df[ses_src].apply(_norm_ses)

    age_col = detect_age_column(df)
    if age_col is None:
        df["age_months"] = pd.NA
    else:
        df["age_months"] = normalize_age_to_months(df[age_col], age_unit=age_unit)

    return df


# -----------------------------------------------------------------------------
# CSV building
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """One dataset configuration."""
    root: Path
    layout: str                # "long" or "cross"
    input_list: Path
    age_unit: str              # "m" or "y"


def extract_sub_ses(path_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract BIDS subject/session from a path by scanning path parts.

    Returns
    -------
    (sub_id, ses_id)
        Each can be None if not found.
    """
    parts = Path(path_str).parts
    sid = next((x for x in parts if x.startswith("sub-")), None)
    ses = next((x for x in parts if x.startswith("ses-")), None)
    return sid, ses


def build_rows(
    spec: DatasetSpec,
    sex_map: Dict[str, int],
    age_map: Dict[str, float],
    sessions_df: Optional[pd.DataFrame],
    dest: Path,
) -> List[Dict[str, object]]:
    """
    Build per-image rows for one dataset.

    Parameters
    ----------
    spec:
        Dataset configuration.
    sex_map, age_map:
        Lookups from participants.tsv.
    sessions_df:
        Sessions dataframe for long layout, else None.
    dest:
        Base destination folder for brain/segm/latent.

    Returns
    -------
    list[dict]
        Rows for the combined CSV.
    """
    ds_name = spec.root.name
    paths = read_paths_list(spec.input_list)

    rows: List[Dict[str, object]] = []

    for p in paths:
        sid, ses = extract_sub_ses(p)
        if not sid:
            continue

        if spec.layout == "cross":
            image_uid = sid
            age = age_map.get(sid, pd.NA)
            base = sid
        else:
            image_uid = ses or sid
            base = f"{sid}_{ses}" if ses else sid

            age = pd.NA
            if sessions_df is not None and ses:
                r = sessions_df[(sessions_df["participant_id"] == sid) & (sessions_df["session_id"] == ses)]
                if not r.empty:
                    age = r.iloc[0].get("age_months", pd.NA)

        sex = sex_map.get(sid, -1)

        rows.append({
            "dataset": ds_name,
            "subject_id": sid,
            "image_uid": image_uid,
            "sex": sex,
            "age": age,  # months (raw, will be copied to age_bef_norm later)
            "image_path": str(dest / f"{base}_brain.nii.gz"),
            "segm_path": str(dest / f"{base}_segm.nii.gz"),
            "latent_path": str(dest / f"{base}_latent.npz"),
        })

    return rows


def assign_stratified_folds(df: pd.DataFrame, folds: int, seed: int) -> pd.Series:
    """
    Assign stratified folds at the *subject level* using sex + age bins.

    Parameters
    ----------
    df:
        Dataframe with columns ``subject_id``, ``sex``, ``age``.
    folds:
        Number of folds.
    seed:
        Random seed.

    Returns
    -------
    pd.Series
        Series mapping subject_id -> split (1..folds).
    """
    subj = (
        df.groupby("subject_id")
          .agg(sex=("sex", "first"), age=("age", "mean"))
          .reset_index()
    )

    # age bins; fill missing with median to avoid dropping subjects
    age_filled = subj["age"].fillna(subj["age"].median())
    n_bins = min(folds, 5)
    subj["age_bin"] = pd.qcut(age_filled, q=n_bins, duplicates="drop").astype(str)

    subj["strata"] = subj["sex"].astype(str) + "_" + subj["age_bin"]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    fold_assign: Dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(skf.split(subj, subj["strata"]), start=1):
        for idx in test_idx:
            fold_assign[subj.iloc[idx]["subject_id"]] = fold

    return df["subject_id"].map(fold_assign)


def normalize_age_inplace(df: pd.DataFrame) -> None:
    """
    Copy raw age into ``age_bef_norm`` and min-max normalize age into ``age``.

    Notes
    -----
    Operates in-place.
    """
    
    df["age_bef_norm"] = df["age"]
    df["age_bef_norm"] = pd.to_numeric(df["age_bef_norm"], errors="coerce").round(3)

    ages = pd.to_numeric(df["age_bef_norm"], errors="coerce")
    min_a, max_a = ages.min(), ages.max()
    if pd.isna(min_a) or pd.isna(max_a) or max_a == min_a:
        # avoid divide-by-zero
        df["age"] = pd.NA
        return

    df["age"] = ((ages - min_a) / (max_a - min_a)).round(4)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    p = argparse.ArgumentParser(
        description="Create combined dataset CSV with stratified folds and normalized age."
    )
    p.add_argument("--bids-roots", nargs="+", required=True, help="List of BIDS root directories.")
    p.add_argument("--layouts", nargs="+", choices=["long", "cross"], required=True,
                   help="Layout per root: long or cross.")
    p.add_argument("--input-lists", nargs="+", required=True,
                   help="List of preprocess_<dataset>.txt files, one per BIDS root.")
    p.add_argument("--age-units", nargs="+", default=None,
                   help='Age units per dataset ("y" years or "m" months). '
                        "If one value is given, it is applied to all datasets.")
    p.add_argument("--dest-path-for-images", required=True,
                   help="Base destination path for brain/segm/latent files.")
    p.add_argument("--out-csv", default="dataset.csv", help="Output CSV file path.")
    p.add_argument("--folds", type=int, default=5, help="Number of stratified folds.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p


def _normalize_units_list(units_arg: Optional[Sequence[str]], n: int, parser: argparse.ArgumentParser) -> List[str]:
    """Normalize --age-units to length n (broadcast if a single unit is provided)."""
    if units_arg is None:
        return ["m"] * n
    units = [u.lower() for u in units_arg]
    if len(units) == 1:
        units = units * n
    if len(units) != n:
        parser.error("Number of --age-units must match --bids-roots (or provide a single value).")
    bad = [u for u in units if u not in ("m", "y")]
    if bad:
        parser.error(f'Invalid age unit(s): {bad}. Use "m" or "y".')
    return units


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

    n = len(args.bids_roots)
    if not (len(args.layouts) == n == len(args.input_lists)):
        parser.error("Number of --bids-roots, --layouts, and --input-lists must match.")

    age_units = _normalize_units_list(args.age_units, n, parser)

    # Build dataset specs
    specs: List[DatasetSpec] = []
    for root, layout, inplist, unit in zip(args.bids_roots, args.layouts, args.input_lists, age_units):
        specs.append(DatasetSpec(
            root=Path(root).resolve(),
            layout=layout,
            input_list=Path(inplist).resolve(),
            age_unit=unit,
        ))

    dest = Path(args.dest_path_for_images).resolve()

    # Load metadata + build rows
    all_rows: List[Dict[str, object]] = []
    layouts_by_ds: Dict[str, str] = {}

    for spec in specs:
        ds = spec.root.name
        layouts_by_ds[ds] = spec.layout

        sex_map, age_map = load_participants(spec.root, age_unit=spec.age_unit)
        sess_df = load_sessions(spec.root, age_unit=spec.age_unit) if spec.layout == "long" else None

        all_rows.extend(build_rows(spec, sex_map, age_map, sess_df, dest))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("No rows produced (check inputs and paths).")

    # Fold assignment at subject level
    df["split"] = assign_stratified_folds(df, folds=args.folds, seed=args.seed)

    # Print fold counts
    subj_counts = df.groupby("split")["subject_id"].nunique().sort_index()
    print("\nSubjects per split:")
    for sp, cnt in subj_counts.items():
        print(f"  split {int(sp)}: {int(cnt)} subjects")

    # Scan counts for datasets that are long
    any_long = any(layout == "long" for layout in layouts_by_ds.values())
    if any_long:
        scan_counts = df.groupby("split").size().sort_index()
        print("\nScans per split (includes longitudinal datasets):")
        for sp, cnt in scan_counts.items():
            print(f"  split {int(sp)}: {int(cnt)} scans")

    # Normalize age
    normalize_age_inplace(df)
    ages = pd.to_numeric(df["age_bef_norm"], errors="coerce")
    print(f"\nmin age (months): {ages.min()}")
    print(f"max age (months): {ages.max()}")

    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(df)} records to {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())