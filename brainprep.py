#!/usr/bin/env python3
"""
brainprep: Preprocessing for pediatric MRI brain data (BIDS derivatives)

This module writes outputs directly into:

    <bids_root>/derivatives/brainprep/sub-*/ses-*/anat/

Pipeline (minimal outputs):
  1) SynthStrip: skull-stripped image + mask (desc-synthstrip, desc-synthstrip_mask)
  2) N4: bias correction on skull-stripped (desc-n4)
  3) Affine registration to template: warped image + warped mask + transform
  4) SynthSeg: segmentation + QC/vol tables
  5) Intensity normalization (WhiteStripe) in template space: desc-intnorm

Sphinx notes:
- Keep code import-safe: do not run pipeline at import.
- Use `main()` for CLI, and guard with `if __name__ == "__main__":`.

"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.typing import Modality
from multiprocessing import Pool
from tqdm import tqdm
from contextlib import redirect_stderr


NPROC = os.cpu_count() or 1

_STAGE_ORDER = {
    "synthstrip": 0,
    "n4": 1,
    "reg": 2,
    "synthseg": 3,
    "intnorm": 4,
}

def should_run_step(step: str, start_from: str) -> bool:
    return _STAGE_ORDER[step] >= _STAGE_ORDER[start_from]

def pick_registration_input(bp: "BrainprepPaths") -> Path:
    """
    Prefer N4 if it exists, else synthstrip, else raw input.
    """
    if bp.n4.exists():
        return bp.n4
    if bp.synthstrip_img.exists():
        return bp.synthstrip_img
    return bp.input

def reg_outputs_exist(bp: "BrainprepPaths") -> bool:
    # Adjust if your transform extension can be .h5 sometimes
    xfm_ok = bp.xfm.exists() or bp.xfm.with_suffix(".h5").exists()
    return bp.warped.exists() and bp.warped_mask.exists() and xfm_ok

# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------
class TeeLogger:
    """Duplicate stdout/stderr to both terminal and a log file."""

    def __init__(self, logfile_path: Path):
        self.terminal = sys.__stdout__
        self.log = open(logfile_path, "w")
        self.closed = False

    def write(self, message: str) -> None:
        self.terminal.write(message)
        if not self.closed:
            self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        if not self.closed:
            self.log.flush()

    def close(self) -> None:
        if not self.closed:
            self.log.close()
            self.closed = True


# -----------------------------------------------------------------------------
# BIDS parsing + naming
# -----------------------------------------------------------------------------
def get_template_name(template_path: str) -> str:
    """Return template basename without .nii/.nii.gz."""
    base = os.path.basename(template_path)
    base = re.sub(r"\.nii(\.gz)?$", "", base)
    return base


def sanitize_entity(label: str) -> str:
    """Sanitize a label for use in BIDS entities (alphanumeric only)."""
    return re.sub(r"[^A-Za-z0-9]+", "", label)


def parse_bids_info(path: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Extract (dataset, sub_id, ses_id, run_id) from a path.

    Parameters
    ----------
    path:
        Path to a BIDS file.

    Returns
    -------
    dataset:
        A heuristic dataset name (based on substrings).
    sub_id:
        e.g. "sub-10001" or None
    ses_id:
        e.g. "ses-001" or None
    run_id:
        e.g. "run-01" or None
    """
    dataset = "unknown_dataset"
    if "hc-bcp" in path:
        dataset = "hc-bcp"
    elif "hc-calgary-preschool" in path:
        dataset = "hc-calgary-preschool"
    elif "hc-new-england" in path:
        dataset = "hc-new-england"
    elif "daufin" in path:
        dataset = "hc-daufin"
    elif "pixar" in path:
        dataset = "hc-pixar"
    elif "ping" in path:
        dataset = "hc-ping"
    elif "mtbi-koala" in path:
        dataset = "mtbi-koala"

    m = re.search(r"(sub-[^/]+)", path)
    sub_id = m.group(1) if m else None

    m = re.search(r"(ses-[^/]+)", path)
    ses_id = m.group(1) if m else None

    m = re.search(r"(run-\d+)", path)
    run_id = m.group(1) if m else None

    return dataset, sub_id, ses_id, run_id

def intnorm_worker(bp: "BrainprepPaths") -> bool:
    """
    Pool worker: WhiteStripe normalize template-space image and mask it.
    Returns True on success, False on failure.
    """
    try:
        if not bp.warped.exists():
            raise RuntimeError(f"Missing warped image: {bp.warped}")
        if not bp.warped_mask.exists():
            raise RuntimeError(f"Missing warped mask: {bp.warped_mask}")

        whitestripe_intnorm(bp.warped, bp.warped_mask, bp.intnorm)
        return True
    except Exception as e:
        print(f"[ERROR][intnorm] {bp.subject_id if hasattr(bp,'subject_id') else ''} {bp.input}: {e}")
        return False


def reg_type_to_desc(registration_type: str) -> str:
    """
    Map ANTs registration type flag to a desc label.

    t -> translation
    r -> rigid
    a -> affine
    """
    rt = (registration_type or "").lower()
    if rt == "t":
        return "translation"
    if rt == "r":
        return "rigid"
    if rt == "a":
        return "affine"
    raise ValueError(f"Unknown registration_type={registration_type!r} (expected: t, r, a)")


@dataclass(frozen=True)
class BrainprepPaths:
    """All output paths for one input image."""
    input: Path
    sub_id: str
    ses_id: Optional[str]          # <- now optional
    run_id: Optional[str]
    space: str

    anat_dir: Path
    work_dir: Path

    synthstrip_img: Path
    synthstrip_mask: Path
    n4: Path

    warped: Path
    warped_mask: Path
    xfm: Path

    dseg: Path
    qc: Path
    vol: Path

    intnorm: Path

def make_brainprep_paths(
    input_nifti: str,
    bids_root: str,
    template_path: str,
    registration_type: str = "a",
) -> BrainprepPaths:
    """
    Create BIDS-derivatives paths under derivatives/brainprep.

    Notes
    -----
    - Keeps run entity if present in input path.
    - `space` label is derived from template basename (sanitized).
    - Adds `desc-<registration>` (translation|rigid|affine) from the warp step onward,
      including SynthSeg outputs and intnorm.
    """
    _, sub_id, ses_id, run_id = parse_bids_info(input_nifti)
    sub_id = sub_id or "sub-UNKNOWN"
    
    # If there is no session in the input path, keep it as None
    ses_id = ses_id or None

    template_base = get_template_name(template_path)
    space = sanitize_entity(template_base)

    reg_desc = reg_type_to_desc(registration_type)

    deriv_root = Path(bids_root) / "derivatives" / "brainprep"
    
    subj_root = deriv_root / sub_id
    if ses_id:
        subj_root = subj_root / ses_id

    anat_dir = subj_root / "anat"
    work_dir = (deriv_root / "work" / sub_id / ses_id) if ses_id else (deriv_root / "work" / sub_id)

    anat_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    run_part = f"_{run_id}" if run_id else ""
    ses_part = f"_{ses_id}" if ses_id else ""
    base = f"{sub_id}{ses_part}{run_part}"

    # --- Pre-template-space steps (keep as-is) ---
    synthstrip_img = anat_dir / f"{base}_desc-synthstrip_T1w.nii.gz"
    synthstrip_mask = anat_dir / f"{base}_desc-synthstrip_mask.nii.gz"
    n4 = anat_dir / f"{base}_desc-n4_T1w.nii.gz"

    # --- Template-space outputs: add desc-{reg_desc} consistently ---
    warped = anat_dir / f"{base}_space-{space}_desc-{reg_desc}_T1w.nii.gz"
    warped_mask = anat_dir / f"{base}_space-{space}_desc-{reg_desc}_mask.nii.gz"

    # Transform file: include reg_desc too (optional but consistent)
    xfm = anat_dir / f"{base}_from-T1w_to-{space}_desc-{reg_desc}_mode-image_xfm.mat"

    # SynthSeg outputs: include reg_desc too
    dseg = anat_dir / f"{base}_space-{space}_desc-{reg_desc}-synthseg_dseg.nii.gz"
    qc   = anat_dir / f"{base}_space-{space}_desc-{reg_desc}-synthseg_qc.tsv"
    vol  = anat_dir / f"{base}_space-{space}_desc-{reg_desc}-synthseg_vol.tsv"


    # Intensity normalization: include reg_desc too
    intnorm = anat_dir / f"{base}_space-{space}_desc-{reg_desc}-intnorm_T1w.nii.gz"

    return BrainprepPaths(
        input=Path(input_nifti),
        sub_id=sub_id,
        ses_id=ses_id,
        run_id=run_id,
        space=space,
        anat_dir=anat_dir,
        work_dir=work_dir,
        synthstrip_img=synthstrip_img,
        synthstrip_mask=synthstrip_mask,
        n4=n4,
        warped=warped,
        warped_mask=warped_mask,
        xfm=xfm,
        dseg=dseg,
        qc=qc,
        vol=vol,
        intnorm=intnorm,
    )

def infer_bids_root(input_path: str) -> str:
    p = Path(input_path).resolve()
    parts = p.parts
    for i, part in enumerate(parts):
        if part.startswith("sub-"):
            print('PATH', str(Path(*parts[:i])))
            return str(Path(*parts[:i]))
    raise ValueError(f"Could not infer BIDS root from: {input_path}")


# -----------------------------------------------------------------------------
# Command runners
# -----------------------------------------------------------------------------
def run(cmd: List[str], *, quiet: bool = False) -> int:
    """
    Run a command. Returns returncode.

    Uses subprocess for safer quoting than os.system.
    """
    if not quiet:
        print("📣", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL if quiet else None,
                          stderr=subprocess.DEVNULL if quiet else None)
    return proc.returncode


def run_synthstrip(inp: Path, out_img: Path, out_mask: Path, *, quiet: bool = True) -> None:
    """Run FreeSurfer SynthStrip."""
    if out_img.exists() and out_mask.exists():
        return
    cmd = ["mri_synthstrip", "-i", str(inp), "-o", str(out_img), "-m", str(out_mask)]
    rc = run(cmd, quiet=quiet)
    if rc != 0 or not out_img.exists():
        raise RuntimeError(f"SynthStrip failed: {inp}")


def run_n4(inp: Path, out: Path, shrink_factor: int, *, quiet: bool = True) -> None:
    """Run ANTs N4 bias-field correction."""
    if out.exists():
        return
    cmd = [
        "N4BiasFieldCorrection", "-d", "3",
        "-i", str(inp),
        "-o", str(out),
        "-s", str(shrink_factor),
        "-v"
    ]
    rc = run(cmd, quiet=quiet)
    if rc != 0 or not out.exists():
        raise RuntimeError(f"N4 failed: {inp}")


def run_registration(moving: Path, fixed: Path, prefix: Path, threads: int, regtype: str, *, quiet: bool = True) -> Tuple[Path, Path]:
    """
    Run antsRegistrationSyNQuick.sh.

    Returns
    -------
    warped_img:
        Path to prefix+'Warped.nii.gz'
    xfm_path:
        Path to either '0GenericAffine.mat' or 'Composite.h5'
    """
    cmd = [
        "antsRegistrationSyNQuick.sh",
        "-d", "3",
        "-f", str(fixed),
        "-m", str(moving),
        "-o", str(prefix),
        "-n", str(threads),
        "-t", str(regtype),
    ]
    rc = run(cmd, quiet=quiet)
    if rc != 0:
        raise RuntimeError("Registration command failed")

    warped = Path(str(prefix) + "Warped.nii.gz")
    if not warped.exists():
        raise RuntimeError("Registration did not produce Warped.nii.gz")

    mat = Path(str(prefix) + "0GenericAffine.mat")
    h5 = Path(str(prefix) + "Composite.h5")
    if mat.exists():
        return warped, mat
    if h5.exists():
        return warped, h5
    raise RuntimeError("Registration produced no affine (.mat) or composite (.h5) transform")

def register_only(
    paths: BrainprepPaths,
    template: Path,
    threads: int,
    regtype: str,
    template_mask: Optional[Path],
    keep_work: bool,
    quiet: bool,
) -> BrainprepPaths:
    """
    Registration-only version of process_one().

    - Does NOT run SynthStrip or N4.
    - Picks the best available moving image:
        n4 -> synthstrip_img -> raw input
    - For mask in template space:
        - if template_mask is provided: use it
        - else: warp synthstrip_mask if it exists
          (if no mask exists, we error because intnorm needs a mask)
    """
    # ---- pick moving image
    if paths.n4.exists():
        moving = paths.n4
    elif paths.synthstrip_img.exists():
        moving = paths.synthstrip_img
    else:
        moving = paths.input

    # ---- registration
    prefix = paths.work_dir / "tmpreg_"
    warped_tmp, xfm_tmp = run_registration(
        moving, template, prefix, threads, regtype, quiet=quiet
    )

    # Move warped to final name
    if not paths.warped.exists():
        shutil.move(str(warped_tmp), str(paths.warped))

    # Move transform to final name (mat or h5)
    xfm_to_apply = paths.xfm
    if xfm_tmp.suffix == ".h5":
        xfm_to_apply = paths.xfm.with_suffix(".h5")

    if not xfm_to_apply.exists():
        shutil.move(str(xfm_tmp), str(xfm_to_apply))

    # ---- mask in template space
    if template_mask is not None and template_mask.exists():
        if not paths.warped_mask.exists():
            try:
                paths.warped_mask.symlink_to(template_mask)
            except Exception:
                shutil.copy2(template_mask, paths.warped_mask)
    else:
        # We need a subject-space mask to warp. Prefer synthstrip mask.
        if not paths.synthstrip_mask.exists():
            raise RuntimeError(
                "Cannot create warped_mask without --template-mask or an existing synthstrip mask. "
                "Either run SynthStrip first or pass --template-mask."
            )
        apply_transform_mask(
            paths.synthstrip_mask, paths.warped, xfm_to_apply, paths.warped_mask, quiet=quiet
        )

    # ---- cleanup
    if not keep_work:
        try:
            for p in paths.work_dir.glob("tmpreg_*"):
                if p.is_file():
                    p.unlink()
        except Exception:
            pass

    return paths

def apply_transform_mask(mask_subj: Path, ref_img: Path, xfm: Path, out_mask: Path, *, quiet: bool = True) -> None:
    """Transform subject-space mask into template space using nearest-neighbor."""
    if out_mask.exists():
        return
    cmd = [
        "antsApplyTransforms", "-d", "3",
        "-i", str(mask_subj),
        "-r", str(ref_img),
        "--interpolation", "NearestNeighbor",
        "-t", str(xfm),
        "-o", str(out_mask),
    ]
    rc = run(cmd, quiet=quiet)
    if rc != 0 or not out_mask.exists():
        raise RuntimeError(f"Failed to transform mask: {mask_subj}")


def whitestripe_intnorm(reg_nii: Path, reg_mask: Path, out_nii: Path) -> None:
    """WhiteStripe normalize a template-space image and apply template-space mask."""
    if out_nii.exists():
        return

    reg_img = nib.load(str(reg_nii))
    reg_arr = reg_img.get_fdata()

    ws = WhiteStripeNormalize()
    normalized = ws(reg_arr, mask=None, modality=Modality.T1)

    mask_arr = nib.load(str(reg_mask)).get_fdata()
    brain = normalized.copy()
    brain[mask_arr == 0.0] = brain.min()

    nib.save(nib.Nifti1Image(brain, reg_img.affine, reg_img.header), str(out_nii))


def run_synthseg_batch(pairs: List[Tuple[Path, Path]], qc_paths: List[Path], vol_paths: List[Path],
                       threads: int, *, quiet: bool = True) -> None:
    """
    Run SynthSeg in multi-image mode.

    Parameters
    ----------
    pairs:
        List of (input_image, output_dseg).
    qc_paths, vol_paths:
        Per-image qc/vol outputs.
    """
    if not pairs:
        return

    tmp_dir = Path(".")

    inp_list = tmp_dir / "temp-input.txt"
    out_list = tmp_dir / "temp-output.txt"
    qc_list = tmp_dir / "temp-qc.txt"
    vol_list = tmp_dir / "temp-vol.txt"

    with inp_list.open("w") as f_in, out_list.open("w") as f_out:
        for inp, out in pairs:
            f_in.write(str(inp) + "\n")
            f_out.write(str(out) + "\n")

    with qc_list.open("w") as f_qc, vol_list.open("w") as f_vol:
        for qc, vol in zip(qc_paths, vol_paths):
            f_qc.write(str(qc) + "\n")
            f_vol.write(str(vol) + "\n")

    cmd = [
        "mri_synthseg",
        "--i", str(inp_list),
        "--o", str(out_list),
        "--vol", str(vol_list),
        "--qc", str(qc_list),
        "--threads", str(threads),
        "--cpu",
    ]
    rc = run(cmd, quiet=quiet)
    if rc != 0:
        raise RuntimeError("SynthSeg failed")

    for p in [inp_list, out_list, qc_list, vol_list]:
        try:
            p.unlink()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Pipeline execution
# -----------------------------------------------------------------------------
def process_one(paths: BrainprepPaths, template: Path, threads: int, shrinkf: int, regtype: str,
                skip_n4: bool, template_mask: Optional[Path], keep_work: bool, quiet: bool) -> BrainprepPaths:
    """
    Run steps 1-3 for one image: SynthStrip -> N4 -> Registration -> mask transform.

    Returns the same `paths` on success.
    """
    # Step 1: SynthStrip
    run_synthstrip(paths.input, paths.synthstrip_img, paths.synthstrip_mask, quiet=quiet)

    # Step 2: N4 (or skip)
    if skip_n4:
        # Treat the skull-stripped image as "n4 output" for downstream steps.
        if not paths.n4.exists():
            # Symlink preferred; fallback to copy.
            try:
                paths.n4.symlink_to(paths.synthstrip_img)
            except Exception:
                shutil.copy2(paths.synthstrip_img, paths.n4)
    else:
        run_n4(paths.synthstrip_img, paths.n4, shrinkf, quiet=quiet)

    # Step 3: Affine registration
    # Write ANTs intermediate products into work_dir
    prefix = paths.work_dir / "tmpreg_"

    warped_tmp, xfm_tmp = run_registration(paths.n4, template, prefix, threads, regtype, quiet=quiet)

    # Move warped to final name
    if not paths.warped.exists():
        shutil.move(str(warped_tmp), str(paths.warped))

    # Move transform to final name (mat or h5)
    xfm_to_apply = paths.xfm
    if xfm_tmp.suffix == ".h5":
        xfm_to_apply = paths.xfm.with_suffix(".h5")

    if not xfm_to_apply.exists():
        shutil.move(str(xfm_tmp), str(xfm_to_apply))

    # Mask in template space
    if template_mask is not None and template_mask.exists():
        # If user supplies a template-space mask, use it directly
        if not paths.warped_mask.exists():
            # symlink or copy
            try:
                paths.warped_mask.symlink_to(template_mask)
            except Exception:
                shutil.copy2(template_mask, paths.warped_mask)
    else:
        apply_transform_mask(paths.synthstrip_mask, paths.warped, xfm_to_apply, paths.warped_mask, quiet=quiet)

    # Cleanup work dir
    if not keep_work:
        try:
            for p in paths.work_dir.glob("tmpreg_*"):
                if p.is_file():
                    p.unlink()
        except Exception:
            pass

    return paths


# def run_pipeline(bids_root: str, inputs: List[str], template: str, threads: int, shrinkf: int, regtype: str,
#                  no_bfc_list: Optional[set], template_mask: Optional[str], keep_work: bool, quiet: bool) -> None:
#     """
#     Full brainprep pipeline: steps 1-5.
#     """
#     template_p = Path(template)
#     tmask_p = Path(template_mask) if template_mask else None

#     # Build per-image paths
#     all_paths: Dict[str, BrainprepPaths] = {}
#     for p in inputs:
#         if not Path(p).exists():
#             print(f"[WARN] missing input: {p}")
#             continue
#         all_paths[p] = make_brainprep_paths(p, bids_root, template)

#     # Steps 1-3 sequential (keeps external calls simpler to debug)
#     ok_paths: List[BrainprepPaths] = []
#     for inp, bp in tqdm(all_paths.items(), desc="SynthStrip/N4/Reg"):
#         try:
#             skip_n4 = (no_bfc_list is not None and inp in no_bfc_list)
#             ok_paths.append(
#                 process_one(
#                     bp, template_p, threads, shrinkf, regtype,
#                     skip_n4=skip_n4,
#                     template_mask=tmask_p,
#                     keep_work=keep_work,
#                     quiet=quiet,
#                 )
#             )
#         except Exception as e:
#             print(f"[ERROR] {inp}: {e}")

#     if not ok_paths:
#         print("[!] No successful registrations; stopping.")
#         return

#     # Step 5 (your order in practice): SynthSeg
#     seg_pairs: List[Tuple[Path, Path]] = []
#     qc_paths: List[Path] = []
#     vol_paths: List[Path] = []

#     for bp in ok_paths:
#         # choose input for synthseg:
#         # - warped is template-space (consistent)
#         inp_img = bp.warped
#         if bp.dseg.exists():
#             continue
#         seg_pairs.append((inp_img, bp.dseg))
#         qc_paths.append(bp.qc)
#         vol_paths.append(bp.vol)

#     if seg_pairs:
#         run_synthseg_batch(seg_pairs, qc_paths, vol_paths, threads=threads, quiet=quiet)

#     # Step 4 (your code runs intnorm in parallel): WhiteStripe + masked brain in template space
#     # parallelize on the original input keys (we pass BrainprepPaths via closure)
    

#     with Pool(processes=threads) as pool:
#         list(tqdm(pool.imap_unordered(intnorm_worker, ok_paths),
#                 total=len(ok_paths), desc="IntensityNorm"))

#     # Final sanity
#     missing = [bp for bp in ok_paths if not bp.intnorm.exists()]
#     if missing:
#         print(f"[WARN] {len(missing)} intnorm outputs missing")
#     print("✅ Finished all steps!")

def run_pipeline(
    inputs: List[str],
    template: str,
    threads: int,
    shrinkf: int,
    regtype: str,
    no_bfc_list: Optional[set],
    template_mask: Optional[str],
    keep_work: bool,
    quiet: bool,
    start_from: str = "synthstrip",     # NEW
    skip_existing: bool = False,        # NEW
) -> None:
    """
    Full brainprep pipeline: steps 1-5.

    Parameters
    ----------
    start_from:
        One of: synthstrip, n4, reg, synthseg, intnorm.
        Use 'reg' to skip SynthStrip/N4 and start at registration.
    skip_existing:
        If True, skip a step when its outputs already exist.
    """
    template_p = Path(template)
    tmask_p = Path(template_mask) if template_mask else None

    # Build per-image paths
    all_paths: Dict[str, BrainprepPaths] = {}
    for p in inputs:
        if not Path(p).exists():
            print(f"[WARN] missing input: {p}")
            continue

        inferred_root = infer_bids_root(p)
        all_paths[p] = make_brainprep_paths(
            p,
            inferred_root,
            template,
            registration_type=regtype,
        )

    ok_paths: List[BrainprepPaths] = []

    # --------------------------
    # Steps 1-3: SynthStrip / N4 / Reg
    # --------------------------
    for inp, bp in tqdm(all_paths.items(), desc="SynthStrip/N4/Reg"):
        try:
            skip_n4 = (no_bfc_list is not None and inp in no_bfc_list)

            # If we start from reg: don't run synthstrip/n4, only ensure registration exists.
            if start_from == "reg":
                # Optionally skip registration if already done
                if skip_existing and reg_outputs_exist(bp):
                    ok_paths.append(bp)
                    continue

                # If you want strict behavior (require N4/synthstrip), enforce it here:
                # if not bp.n4.exists(): raise RuntimeError("Missing N4 output; cannot start-from reg.")

                # Run ONLY registration (you need a function that does reg only)
                # If process_one() always runs synthstrip/n4 too, create a light wrapper:
                bp = register_only(
                    paths=bp,
                    template=template_p,
                    threads=threads,
                    regtype=regtype,
                    template_mask=tmask_p,
                    keep_work=keep_work,
                    quiet=quiet,
                )

                ok_paths.append(bp)
                continue

            # Otherwise: run your original process_one (SynthStrip+N4+Reg),
            # but add skip-existing behavior inside process_one (recommended),
            # OR do coarse skipping here:
            if skip_existing and reg_outputs_exist(bp):
                # already fully registered -> accept it
                ok_paths.append(bp)
                continue

            ok_paths.append(
                process_one(
                    bp, template_p, threads, shrinkf, regtype,
                    skip_n4=skip_n4,
                    template_mask=tmask_p,
                    keep_work=keep_work,
                    quiet=quiet,
                    # If you can modify process_one: pass start_from/skip_existing down:
                    # start_from=start_from,
                    # skip_existing=skip_existing,
                )
            )

        except Exception as e:
            print(f"[ERROR] {inp}: {e}")

    if not ok_paths:
        print("[!] No successful registrations; stopping.")
        return

    # --------------------------
    # SynthSeg (batch)
    # --------------------------
    seg_pairs: List[Tuple[Path, Path]] = []
    qc_paths: List[Path] = []
    vol_paths: List[Path] = []

    if should_run_step("synthseg", start_from):
        for bp in ok_paths:
            if not should_run_step("synthseg", start_from):
                break

            if skip_existing and bp.dseg.exists() and bp.qc.exists() and bp.vol.exists():
                continue

            seg_pairs.append((bp.warped, bp.dseg))
            qc_paths.append(bp.qc)
            vol_paths.append(bp.vol)

        if seg_pairs:
            run_synthseg_batch(seg_pairs, qc_paths, vol_paths, threads=threads, quiet=quiet)

    # --------------------------
    # Intensity norm (parallel)
    # --------------------------
    if should_run_step("intnorm", start_from):
        # Filter out those already done if resume mode
        todo = ok_paths
        if skip_existing:
            todo = [bp for bp in ok_paths if not bp.intnorm.exists()]

        if todo:
            with Pool(processes=threads) as pool:
                # If your intnorm_worker needs template_mask, you can wrap it or make it global.
                list(tqdm(pool.imap_unordered(intnorm_worker, todo),
                          total=len(todo), desc="IntensityNorm"))

    # Final sanity
    missing = [bp for bp in ok_paths if should_run_step("intnorm", start_from) and not bp.intnorm.exists()]
    if missing:
        print(f"[WARN] {len(missing)} intnorm outputs missing")
    print("✅ Finished all steps!")



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def read_inputs_list(txt_path: str) -> List[str]:
    """Read one path per line, removing enclosing quotes if present."""
    out: List[str] = []
    with open(txt_path, "r") as f:
        for line in f:
            p = line.strip()
            if p.startswith('"') and p.endswith('"'):
                p = p[1:-1]
            if p:
                out.append(p)
    return out


def read_no_bfc_list(txt_path: Optional[str]) -> Optional[set]:
    """Read a set of paths to skip N4."""
    if not txt_path:
        return None
    p = Path(txt_path)
    if not p.exists():
        return None
    with p.open("r") as f:
        return set(line.strip() for line in f if line.strip())


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="brainprep: pediatric MRI preprocessing (BIDS derivatives)")

    parser.add_argument("--inputs", required=True, help="Text file with paths to T1w images (one per line)")
    parser.add_argument("--template", required=True, help="Template image for antsRegistrationSyNQuick.sh")
    # parser.add_argument("--bids-root", required=True, help="BIDS dataset root (contains sub-*/ and derivatives/)")

    parser.add_argument("-t", "--threads", type=int, default=NPROC, help="Threads / processes (default: all cores)")
    parser.add_argument("-s", "--shrink-factor", type=int, default=4, help="N4 shrink factor (default=4)")
    parser.add_argument("-r", "--registration-type", type=str, default="a", help="ANTs registration type: t=translation, r=rigid, a=affine (default: a).")

    parser.add_argument("--no-bfc", type=str, default=None,
                        help="Text file with input paths for which to skip N4 (bias correction)")
    parser.add_argument("--template-mask", type=str, default=None,
                        help="Optional template-space brain mask (used instead of transforming subject mask)")
    parser.add_argument("--keep-work", action="store_true", help="Keep intermediate ANTs work files")
    parser.add_argument("--quiet", action="store_true", help="Suppress tool stdout/stderr")

    parser.add_argument("--dataset", type=str, default="dataset", help="Used for log filename only")
    parser.add_argument("--start-from", choices=list(_STAGE_ORDER.keys()),
                    default="synthstrip",
                    help="Start pipeline from a given stage (synthstrip|n4|reg|synthseg|intnorm).")
    parser.add_argument("--skip-existing", action="store_true",
                    help="Skip steps whose outputs already exist.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    args = build_argparser().parse_args(argv)

    inputs = read_inputs_list(args.inputs)
    no_bfc_set = read_no_bfc_list(args.no_bfc)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # e.g., 2026-02-12_14-03-59
    logfile = Path(f"preprocessing_log_{args.dataset}_{ts}.txt")
    tee = TeeLogger(logfile)

    start_time = time.time()
    sys.stdout = tee
    with redirect_stderr(tee):
        try:
            run_pipeline(
                inputs=inputs,
                template=args.template,
                threads=args.threads,
                shrinkf=args.shrink_factor,
                regtype=args.registration_type,
                no_bfc_list=no_bfc_set,
                template_mask=args.template_mask,
                keep_work=args.keep_work,
                start_from=args.start_from,
                skip_existing=args.skip_existing,
                quiet=args.quiet,
            )
            end_time = time.time()
            elapsed_minutes = (end_time - start_time) / 60.0
            print(f"🕒 Total time: {elapsed_minutes:.2f} minutes")
        finally:
            sys.stdout = sys.__stdout__
            tee.close()

    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
