#!/usr/bin/env python

import os
import re
import argparse
import time
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm
import sys
from contextlib import redirect_stderr

from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
from intensity_normalization.typing import Modality

NPROC = os.cpu_count()

def final_normalize_brain(input_path):
    """
    1) WhiteStripe on the registered volume
    2) Use the registered mask to zero out outside for final "brain"
    3) Save in step5 as sub-XXX_ses-YYY_normalized.nii.gz, sub-XXX_ses-YYY_brain.nii.gz
    """
    sp = outputs_dict[input_path]
    reg_nii  = sp["registered"]       # template-space volume
    # if args.template_mask and "nihpd" in template:
    #     mask_nii = args.template_mask
    # else:
    mask_nii = sp["registered_mask"] # template-space mask
    norm_nii = sp["normalized"]
    brain_nii= sp["brain"]

    # if os.path.exists(norm_nii) and os.path.exists(brain_nii):
    #     return

    try:
        reg_img = nib.load(reg_nii)
        reg_arr = reg_img.get_fdata()

        # WhiteStripe
        if not os.path.exists(norm_nii):
            ws = WhiteStripeNormalize()
            # No mask => global approach
            # if args.template_mask and "nihpd" in template:
            #     mask_arr = nib.load(mask_nii).get_fdata()
            #     normalized_arr = ws(reg_arr, mask=mask_arr, modality=Modality.T1)
            #     nib.save(nib.Nifti1Image(normalized_arr, reg_img.affine, reg_img.header), norm_nii)
            # else:
            print('\n Saving normalized array...')
            normalized_arr = ws(reg_arr, mask=None, modality=Modality.T1)
            nib.save(nib.Nifti1Image(normalized_arr, reg_img.affine, reg_img.header), norm_nii)
            # Brain extraction using the template-space mask
            print('\n Doing brain extraction...')
            mask_arr = nib.load(mask_nii).get_fdata()
            brain_arr = normalized_arr.copy()
            brain_arr[mask_arr == 0.] = brain_arr.min()
            nib.save(nib.Nifti1Image(brain_arr, reg_img.affine, reg_img.header), brain_nii)
        else:
            normalized_arr = nib.load(norm_nii).get_fdata()

            # Brain extraction using the template-space mask
            print('\n Doing brain extraction...')
            mask_arr = nib.load(mask_nii).get_fdata()
            brain_arr = normalized_arr.copy()
            brain_arr[mask_arr == 0.] = brain_arr.min()
            nib.save(nib.Nifti1Image(brain_arr, reg_img.affine, reg_img.header), brain_nii)

    except Exception as e:
        print(f"[ERROR] Final normalization failed for {input_path}: {e}")

class TeeLogger:
    def __init__(self, logfile_path):
        self.terminal = sys.__stdout__
        self.log = open(logfile_path, "w")
        self.closed = False

    def write(self, message):
        self.terminal.write(message)
        if not self.closed:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.closed:
            self.log.flush()

    def close(self):
        if not self.closed:
            self.log.close()
            self.closed = True

def get_template_name(template_path):
    base = os.path.basename(template_path)
    base = re.sub(r'\.nii(\.gz)?$', '', base)  # remove .nii or .nii.gz
    return base

################################################################################
# HELPER: parse sub-XXX/ses-YYY from a path, plus dataset
################################################################################

def parse_bids_info(path):
    """
    Attempt to extract dataset, sub_id, ses_id, run_id from the path.
    We look for 'hc-bcp', 'hc-calgary-preschool' or fallback to 'unknown_dataset'.
    Then we parse sub-XXX, ses-YYY from the path or filename.
    """
    dataset = "unknown_dataset"
    if "hc-bcp" in path:
        dataset = "hc-bcp"
    elif "hc-calgary-preschool" in path:
        dataset = "hc-calgary-preschool"
    elif "daufin" in path:
        dataset = "hc-daufin"
    elif "ping" in path:
        dataset = "hc-ping"
    elif "mtbi-koala" in path:
        dataset = "mtbi-koala"

    m = re.search(r"(sub-[^/]+)", path)
    if m:
        sub_id = m.group(1)    # gives "sub-CTL_01"
    else:
        sub_id = None

    m = re.search(r"(ses-[^/]+)", path)
    ses_id = m.group(1) if m else None

    m = re.search(r"(run-[^/]+)", path)
    run_id = m.group(1) if m else None

    return dataset, sub_id, ses_id, run_id


################################################################################
# 1) Build a function that returns the BIDS-like path for each pipeline step
################################################################################

def get_step_paths(input_nifti, preproc_dir, template_name=None):
    """
    Return a dictionary of file paths for each step, placed in a BIDS-like structure:
      preproc_dir/derivatives/<template_name>/<dataset>/(01_n4, 02_synthstrip, 03_affine_registration, 04_synthseg, 05_intensity_normalization)/sub-XXX/ses-YYY/anat/.

    We'll parse dataset, sub_id, ses_id from input_nifti.
    """
    template_name = template_name or "default_template"

    dataset, sub_id, ses_id, run_id = parse_bids_info(input_nifti)

    # fallback if we can't parse sub/ses
    if not sub_id: sub_id = "sub-UNKNOWN"
    if not ses_id: ses_id = "ses-UNKNOWN"

    deriv_root = os.path.join(preproc_dir, "derivatives", template_name, dataset)

    if dataset == "mtbi-koala" or "hc-ping" or "hc-daufin":
        step1_dir = os.path.join(deriv_root, "01_n4",                sub_id, "anat")
        step2_dir = os.path.join(deriv_root, "02_synthstrip",        sub_id, "anat")
        step3_dir = os.path.join(deriv_root, "03_affine_registration", sub_id, "anat")
        step4_dir = os.path.join(deriv_root, "04_synthseg",          sub_id, "anat")
        step5_dir = os.path.join(deriv_root, "05_intensity_normalization", sub_id, "anat")

        for d in [step1_dir, step2_dir, step3_dir, step4_dir, step5_dir]:
            os.makedirs(d, exist_ok=True)

        # file naming
        corrected_nifti     = os.path.join(step1_dir, f"{sub_id}_corrected.nii.gz")
        skullstrip_nifti    = os.path.join(step2_dir, f"{sub_id}_skullstrip.nii.gz")
        skullstrip_mask_subj = os.path.join(step2_dir, f"{sub_id}_skullstrip_mask.nii.gz")
        resampled_skull = os.path.join(step2_dir, f"{sub_id}_resampled_skull.nii.gz")
        registered_nifti    = os.path.join(step3_dir, f"{sub_id}_turboprep_Warped.nii.gz")
        registered_mask     = os.path.join(step3_dir, f"{sub_id}_mask.nii.gz")
        ants_prefix         = os.path.join(step3_dir, f"{sub_id}_turboprep_")
        synthseg_nifti      = os.path.join(step4_dir, f"{sub_id}_segm.nii.gz")

        # final step 5 outputs
        normalized_nifti    = os.path.join(step5_dir, f"{sub_id}_normalized.nii.gz")
        brain_nifti         = os.path.join(step5_dir, f"{sub_id}_brain.nii.gz")

        return {
            "dataset":  dataset,
            "sub_id":   sub_id,
            "ses_id":   ses_id,
            "input":    input_nifti,
            "corrected": corrected_nifti,
            "skullstrip": skullstrip_nifti,
            "skullstrip_mask_subj": skullstrip_mask_subj,  # subject-space mask from SynthStrip
            "resampled_skull": resampled_skull,
            "registered": registered_nifti,
            "registered_mask": registered_mask,            # the mask in template space
            "ants_prefix": ants_prefix,
            "synthseg":   synthseg_nifti,
            "normalized": normalized_nifti,
            "brain":      brain_nifti,
        }
    else:
        step1_dir = os.path.join(deriv_root, "01_n4",                sub_id, ses_id, "anat")
        step2_dir = os.path.join(deriv_root, "02_synthstrip",        sub_id, ses_id, "anat")
        step3_dir = os.path.join(deriv_root, "03_affine_registration", sub_id, ses_id, "anat")
        step4_dir = os.path.join(deriv_root, "04_synthseg",          sub_id, ses_id, "anat")
        step5_dir = os.path.join(deriv_root, "05_intensity_normalization", sub_id, ses_id, "anat")


        for d in [step1_dir, step2_dir, step3_dir, step4_dir, step5_dir]:
            os.makedirs(d, exist_ok=True)

        # file naming
        corrected_nifti     = os.path.join(step1_dir, f"{sub_id}_{ses_id}_corrected.nii.gz")
        skullstrip_nifti    = os.path.join(step2_dir, f"{sub_id}_{ses_id}_skullstrip.nii.gz")
        skullstrip_mask_subj= os.path.join(step2_dir, f"{sub_id}_{ses_id}_skullstrip_mask.nii.gz")
        resampled_skull = os.path.join(step2_dir, f"{sub_id}_resampled_skull.nii.gz")
        registered_nifti    = os.path.join(step3_dir, f"{sub_id}_{ses_id}_turboprep_Warped.nii.gz")
        registered_mask     = os.path.join(step3_dir, f"{sub_id}_{ses_id}_mask.nii.gz")
        ants_prefix         = os.path.join(step3_dir, f"{sub_id}_{ses_id}_turboprep_")
        synthseg_nifti      = os.path.join(step4_dir, f"{sub_id}_{ses_id}_segm.nii.gz")

        # final step 5 outputs
        normalized_nifti    = os.path.join(step5_dir, f"{sub_id}_{ses_id}_normalized.nii.gz")
        brain_nifti         = os.path.join(step5_dir, f"{sub_id}_{ses_id}_brain.nii.gz")

        return {
            "dataset":  dataset,
            "sub_id":   sub_id,
            "ses_id":   ses_id,
            "input":    input_nifti,
            "corrected": corrected_nifti,
            "skullstrip": skullstrip_nifti,
            "skullstrip_mask_subj": skullstrip_mask_subj,  # subject-space mask from SynthStrip
            "resampled_skull": resampled_skull,
            "registered": registered_nifti,
            "registered_mask": registered_mask,            # the mask in template space
            "ants_prefix": ants_prefix,
            "synthseg":   synthseg_nifti,
            "normalized": normalized_nifti,
            "brain":      brain_nifti,
        }


################################################################################
# 2) Main
################################################################################

if __name__ == "__main__":
     
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', required=True,
                        help='text file with paths to T1w images (one per line)')
    parser.add_argument('--template', required=True,
                        help='template image for antsRegistrationSyNQuick.sh')
    parser.add_argument('-m', '--modality', default='t1', help='Modality for WhiteStripe')
    parser.add_argument('-t', '--threads', type=int, default=NPROC, help='threads (default: all cores)')
    parser.add_argument('-s', '--shrink-factor', type=int, default=4, help='N4 shrink factor (default=4)')
    parser.add_argument('-r', '--registration-type', type=str, default='a',
                        help='ANTS reg type {t,r,a} (default=a=affine)')
    parser.add_argument('--no-bfc', type=str,
                        help='text file with input paths for which to skip bias field correction')
    parser.add_argument('--keep', action='store_true',
                        help='Keep intermediate files')
    parser.add_argument('--preproc-dir', default='preproces_bcp_cp',
                        help='Top-level folder for derivatives (default=preproces_bcp_cp)')
    parser.add_argument('--template-mask', type=str, default=None,
                    help='Optional template-space brain mask (used instead of transforming subject mask)')
    parser.add_argument('--dataset', type=str, default=None,
                    help='Name of dataset to be preprocessed')


    args = parser.parse_args()
    inp_file = args.inputs
    template = args.template

    threads  = args.threads
    shrinkf  = args.shrink_factor
    regtype  = args.registration_type
    keepint  = args.keep
    preproc_dir = args.preproc_dir
    dataset_name = args.dataset
    
    logfile_path = f"preprocessing_log_{dataset_name}.txt"
    tee = TeeLogger(logfile_path)

    sys.stdout = tee
    with redirect_stderr(tee):
        try:
            # if args.template_mask and "nihpd" in template:
            #     print(f"📌 Using non-skull-stripped image for registration and template mask for brain extraction.")

            # read input lines
            with open(inp_file, 'r') as f:
                inp_list = []
                for line in f:
                    p = line.strip()
                    # remove leading + trailing quotes if present
                    if p.startswith('"') and p.endswith('"'):
                        p = p[1:-1]
                    inp_list.append(p)


            nbc_list = set()
            if args.no_bfc and os.path.exists(args.no_bfc):
                with open(args.no_bfc, 'r') as f:
                    nbc_list = set(l.strip() for l in f.readlines())

            # build a dictionary: { input_path -> step_paths }
            outputs_dict = {}
            for input_path in inp_list:
                if not os.path.exists(input_path):
                    print(f"Input file does not exist: {input_path}")
                    continue

                template_name = get_template_name(template)
                spaths = get_step_paths(input_path, preproc_dir, template_name)

                # skip bias correction if in no-bfc
                if input_path in nbc_list:
                    spaths["corrected"] = input_path

                outputs_dict[input_path] = spaths

            ############################################################################
            # Step 1) N4 Bias Field Correction + Step 2) SynthStrip (with -m) + Step 3) Registration
            ############################################################################
            for input_path, paths in tqdm(outputs_dict.items(), desc="Bias/SkullStrip/Reg"):
                corr_nii   = paths["corrected"]
                skull_nii  = paths["skullstrip"]
                resampled_skull = paths["resampled_skull"]
                subj_mask  = paths["skullstrip_mask_subj"]
                reg_nii    = paths["registered"]
                mask_nii   = paths["registered_mask"]  # final mask in template space
                prefix     = paths["ants_prefix"]

                # skip if registration output already exists
                # if os.path.exists(reg_nii):
                #     continue

                # BIAS-FIELD CORRECTION
                if corr_nii != input_path and not os.path.exists(corr_nii):
                    cmd = (f'N4BiasFieldCorrection -d 3 -i "{input_path}" '
                        f'-o "{corr_nii}" '
                        f'-s {shrinkf} -v > /dev/null')
                    os.system(cmd)
                if not os.path.exists(corr_nii):
                    print(f"[ERROR] N4 failed for {input_path}")
                    continue

                # SynthStrip with a mask output
                if not os.path.exists(skull_nii):
                    os.system(f'mri_synthstrip -i "{corr_nii}" -o {skull_nii} -m {subj_mask}')#--gpu > /dev/null')
                if not os.path.exists(skull_nii):
                    print(f"[ERROR] SynthStrip failed for {input_path}")
                    continue


                # Registration to template
                # if "nihpd" in template:
                #     # 1) resample moving → template
                #     os.system(
                #         f"antsApplyTransforms \
                #         -d 3 \
                #         -i {skull_nii} \
                #         -r {template} \
                #         -o {resampled_skull} \
                #         --interpolation Linear"
                #         )
                #     reg_cmd = (
                #             f"antsRegistration "
                #             f"--collapse-output-transforms 1 "
                #             f"--dimensionality 3 "
                #             f"--winsorize-image-intensities [0.005,0.995] "
                #             f"--initial-moving-transform [{template},{resampled_skull},1] "
                #             f"--initialize-transforms-per-stage 0 "
                #             f"--interpolation Linear "
                #             f"--output [{prefix}, {prefix}Warped.nii.gz] "
                #             f"--transform Rigid[0.2] "
                #             f"--metric Mattes[{template},{resampled_skull},1,32,Regular,0.3] "
                #             f"--convergence [500x250x100,1e-6,10] "
                #             f"--shrink-factors 6x4x2 "
                #             f"--smoothing-sigmas 4x2x1vox "
                #             f"--use-histogram-matching 1 "
                #             f"--transform Affine[0.1] "
                #             f"--metric Mattes[{template},{resampled_skull},1,32,Regular,0.3] "
                #             f"--convergence [500x250x100,1e-6,10] "
                #             f"--shrink-factors 6x4x2 "
                #             f"--smoothing-sigmas 4x2x1vox "
                #             f"--use-histogram-matching 1 "
                #             f"--write-composite-transform 1"
                #         )

                # else:
                #     reg_cmd = (f'antsRegistrationSyNQuick.sh -d 3 '
                #             f'-f {template} -m {skull_nii} '
                #             f'-o {prefix} -n {threads} -t {regtype} > /dev/null')
                reg_cmd = (f'antsRegistrationSyNQuick.sh -d 3 '
                            f'-f {template} -m {skull_nii} '
                            f'-o {prefix} -n {threads} -t {regtype} > /dev/null')
                # print("📣 Full registration command:")
                # print(reg_cmd)
                print('\n Doing registration...')
                os.system(reg_cmd)

                print(prefix)
                warped_nii = prefix + "Warped.nii.gz"
                print(warped_nii)
                if not os.path.exists(warped_nii):
                    print(f"[ERROR] Registration failed for {input_path}")
                    continue

                # rename warped => reg_nii
                os.rename(warped_nii, reg_nii)

                # rename matrix
                mat_path = prefix + "0GenericAffine.mat"
                h5_path = prefix + "Composite.h5"  # or sometimes just "composite.h5"

                if os.path.exists(mat_path):
                    new_mat = os.path.join(os.path.dirname(prefix), "affine_transf.mat")
                    os.rename(mat_path, new_mat)

                elif os.path.exists(h5_path):
                    new_mat = os.path.join(os.path.dirname(prefix), "composite_transform.h5")
                    os.rename(h5_path, new_mat)
                    print(f"[INFO] Using composite transform: {new_mat}")

                else:
                    print(f"[WARN] No affine (.mat) or composite (.h5) transform found for {prefix}")

                # remove inverse warped if present
                iw = prefix + "InverseWarped.nii.gz"
                if os.path.exists(iw):
                    os.remove(iw)

                # optionally remove intermediate
                if not keepint:
                    # remove skullstrip, etc. as needed
                    pass

                # Now transform subject-space mask -> template space if no template mask exists
                # do nearest-neighbor to preserve 0/1
                # if args.template_mask and "nihpd" in template:
                #     # Use provided template mask — skip transformation
                #     paths["registered_mask"] = os.path.abspath(args.template_mask)
                # else:
                print('Transforming subject-space mask to template space')
                # Transform subject-space mask to template space
                mat_app = (f"antsApplyTransforms -d 3 "
                        f"-i {subj_mask} "
                        f"-r {reg_nii} "
                        f"--interpolation NearestNeighbor "
                        f"-t {new_mat} "
                        f"-o {mask_nii}")
                os.system(mat_app)
                if not os.path.exists(mask_nii):
                    print(f"[ERROR] failed to transform mask for {input_path}")


            ############################################################################
            # Step 4) WhiteStripe Normalization + Brain using Registered Mask
            #         (Use a pool to process in parallel)
            ############################################################################

            # gather all input_paths that remain
            final_list = list(outputs_dict.keys())

            with Pool(processes=threads) as pool:
                for _ in tqdm(pool.imap_unordered(final_normalize_brain, final_list), total=len(final_list)):
                    pass

            ############################################################################
            # Step 5) Optional: SynthSeg on Skull-Stripped (if NIH-PD) or Registered
            ############################################################################

            reg_seg_pairs = []
            for inp, sp in outputs_dict.items():
                # if os.path.exists(sp["synthseg"]):
                #     continue  # Skip if already done

                # Use skull-stripped input if NIH-PD template and brain.nii.gz exists
                # if args.template_mask and "nihpd" in template and os.path.exists(sp["brain"]):
                #     input_img = sp["brain"]
                # else:
                input_img = sp["registered"]

                reg_seg_pairs.append((input_img, sp["synthseg"]))

            if len(reg_seg_pairs) > 0:
                with open("temp-input.txt", "w") as f_in, open("temp-output.txt", "w") as f_out:
                    for r, s in reg_seg_pairs:
                        f_in.write(r + "\n")
                        f_out.write(s + "\n")
                with open("temp-qc.txt", "w") as f_qc, open("temp-vol.txt", "w") as f_vol:
                    for r, s in reg_seg_pairs:
                        vol_csv = s.replace("_segm.nii.gz", "_vol.csv")
                        f_vol.write(vol_csv + "\n")
                        qc_csv = s.replace("_segm.nii.gz", "_qc.csv")
                        f_qc.write(qc_csv + "\n")

                # Run SynthSeg (multiple images mode)
                os.system(f'mri_synthseg --i temp-input.txt --o temp-output.txt --vol temp-vol.txt --qc temp-qc.txt --threads {threads} --cpu')
                os.remove("temp-input.txt")
                os.remove("temp-output.txt")
                os.remove("temp-vol.txt")
                os.remove("temp-qc.txt")

            # Check success
            for inp, sp in list(outputs_dict.items()):
                if not os.path.exists(sp["synthseg"]):
                    print(f"[WARN] SynthSeg failed for {inp}")
                    del outputs_dict[inp]


            print("✅ Finished all steps!")

            end_time = time.time()
            elapsed_minutes = (end_time - start_time) / 60
            print(f"🕒 Total time: {elapsed_minutes:.2f} minutes")

        finally:
            sys.stdout = sys.__stdout__
            tee.close()
