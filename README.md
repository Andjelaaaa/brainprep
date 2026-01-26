# brainprep 🧠
Preprocessing for pediatric (1-7 yo) MRI brain data

This preprocessing follows the BIDS standard and writes all outputs directly into your dataset’s `derivatives/brainprep/` folder. It works with any BIDS-compliant dataset that provides an `exclude.yaml` file (as described below). After raw-image QC, the workflow runs in three steps and supports both cross-sectional and longitudinal datasets.


---

## Install `brainprep` (using `environment.yml`)

1) **Create the conda environment**
```bash
conda env create -f environment.yml
```

2. **Activate it**

```bash
conda activate brainprep
```

3. **Run a script**

```bash
python brainprep.py --help
```


## 1) Create the input list (`create_input_txt.py`)

This step scans one or more **BIDS root** directories and writes a text file containing the **absolute paths** to images that should be preprocessed by `brainprep.py`.

<details>
<summary><strong>📌 Additional instructions (exclude.yaml, layouts, age filters, examples): click the triangle to expand</strong></summary>


### Where `exclude.yaml` lives (default)

By default the script looks for:

```
<bids_root>/code/qc/raw/exclude.yaml
```

You can change the filename with `-e/--exclude-file`, but the folder is expected to be `code/qc/raw/` under each BIDS root.

### What the output file contains

The output is a plain text file (e.g., `to_preprocess_hc-calgary-preschool.txt`) with **one quoted absolute path per line**, like:

```
"/path/to/bids/sub-10001/ses-001/anat/sub-10001_ses-001_T1w.nii.gz"
"/path/to/bids/sub-10002/ses-002/anat/sub-10002_ses-002_T1w.nii.gz"
...
```

This list is later passed to `brainprep.py` via `--inputs`.

---

### `exclude.yaml` format

`exclude.yaml` can be either:

**(A) a YAML list**

```yaml
- sub-10001
- sub-10002_ses-001
- sub-10003_ses-002_run-02
```

**(B) a dict with a list under one of these keys:** `exclude`, `exclude_paths`, `exclude_images`

```yaml
exclude:
  - sub-10001
  - sub-10002_ses-001
  - sub-10003_ses-002_run-02
```

#### What entries mean

You can exclude at different levels:

* `sub-10001`
  → excludes **all sessions/files** for that subject

* `sub-10001_ses-001`
  → excludes **all files** under that session

* `sub-10001_ses-001_run-02`
  → excludes **that specific run** (for `T1w`/`T2w`, this targets `.../anat/<id>_<modality>.nii.gz`)

* You can also provide **path-like patterns / globs**, e.g.

  * `sub-10001/ses-001/**`
  * `sub-*/ses-*/anat/*_T1w.nii.gz`

Notes:

* Excludes are matched against the **relative path** from the dataset root.
* Run-level exclusions only trigger when the exclude string contains `_run-`.

---

### Longitudinal vs cross-sectional layouts

You must specify a layout for each dataset root:

* `-l long` expects: `sub-*/ses-*/anat/*_T1w.nii.gz`
* `-l cross` expects: `sub-*/anat/*_T1w.nii.gz`

You can pass **multiple datasets** at once and give one layout per dataset.

---

### Optional filters

#### Subject filter

Restrict to specific subjects:

* `--subjects sub-10001 sub-10010`
* or `--subjects-file subjects.txt` (one subject per line)

#### Age filter (in months)

You can filter sessions by age:

* `--min-age-months 12 --max-age-months 84`

Age is read from a TSV (tab or CSV is auto-detected):

* `--age-tsv <path>`

  * If not provided: defaults to `<bids_root>/participants.tsv`
* Column names are configurable:

  * `--age-pid-col participant_id`
  * `--age-ses-col session` (used only for longitudinal layout)
  * `--age-col age`
  * `--age-units years|months` (default: years)

Behavior:

* If age is missing / not parseable → the session is **excluded** (conservative).
* For longitudinal datasets, age is looked up by `(participant_id, session)`.
* If a session label ends with `mo` (e.g., `ses-24mo`) and age is not found in TSV, it can be used as a fallback.

---

### Common commands

**Single longitudinal dataset (T1w), using default exclude.yaml location**

```bash
python create_input_txt.py /home/andjela/joplin-intra-inter/hc-calgary-preschool \
  -l long \
  --modality T1w \
  -o to_preprocess_hc-calgary-preschool.txt
```

**With age range (1–7 years)**

```bash
python create_input_txt.py /home/andjela/joplin-intra-inter/hc-calgary-preschool \
  -l long \
  --modality T1w \
  --min-age-months 12 --max-age-months 84 \
  --age-tsv /home/andjela/joplin-intra-inter/hc-calgary-preschool/participants.tsv \
  --age-col age --age-units years \
  -o to_preprocess_hc-calgary-preschool_12to84mo.txt
```

**Multiple datasets at once**

```bash
python create_input_txt.py \
  /home/andjela/joplin-intra-inter/hc-bcp \
  /home/andjela/joplin-intra-inter/hc-calgary-preschool \
  -l long long \
  --modality T1w \
  -o to_preprocess_all.txt
```

**Non-anat modality (example: dwi), using a recursive pattern**

```bash
python create_input_txt.py /path/to/bids \
  -l long \
  --modality dwi \
  -o to_preprocess_dwi.txt
```


</details>

---

## 2) Run the preprocessing (`brainprep.py`)

This step reads the image list produced by `create_input_txt.py` and writes BIDS-derivatives outputs directly into the dataset:

```
<bids_root>/derivatives/brainprep/sub-*/ses-*/anat/
```

### Pipeline steps and outputs

For each input `sub-*/ses-*/anat/*_T1w.nii.gz`, the following steps are applied:

1. **SynthStrip (brain extraction)**

* Tool: `mri_synthstrip`
* Outputs:

  * `sub-*_ses-*_desc-synthstrip_T1w.nii.gz`
  * `sub-*_ses-*_desc-synthstrip_mask.nii.gz`

2. **N4 bias-field correction**

* Tool: `N4BiasFieldCorrection`
* Input: skull-stripped image from SynthStrip
* Output:

  * `sub-*_ses-*_desc-n4_T1w.nii.gz`
* Optional skip: if the input path is listed in `--no-bfc`, the N4 output is replaced by a link/copy of the SynthStrip image.

3. **Affine registration to a template**

* Tool: `antsRegistrationSyNQuick.sh`
* Input: N4 output
* Outputs (template space):

  * `sub-*_ses-*_space-<TEMPLATE>_desc-affine_T1w.nii.gz`
  * Transform:

    * `sub-*_ses-*_from-T1w_to-<TEMPLATE>_mode-image_xfm.mat`
    * or (if ANTs outputs a composite transform):

      * `sub-*_ses-*_from-T1w_to-<TEMPLATE>_mode-image_xfm.h5`
* Mask handling:

  * If `--template-mask` is provided, it is used directly as the template-space mask.
  * Otherwise the subject mask is transformed into template space using `antsApplyTransforms`:

    * `sub-*_ses-*_space-<TEMPLATE>_desc-affine_mask.nii.gz`

4. **SynthSeg segmentation (template space)**

* Tool: `mri_synthseg` (batch mode)
* Input: registered image in template space
* Outputs:

  * `sub-*_ses-*_space-<TEMPLATE>_desc-synthseg_dseg.nii.gz`
  * `sub-*_ses-*_space-<TEMPLATE>_desc-synthseg_qc.tsv`
  * `sub-*_ses-*_space-<TEMPLATE>_desc-synthseg_vol.tsv`

5. **Intensity normalization (template space)**

* Method: WhiteStripe (Python `intensity_normalization`)
* Input: registered image + template-space mask
* Output:

  * `sub-*_ses-*_space-<TEMPLATE>_desc-intnorm_T1w.nii.gz`

> `<TEMPLATE>` is derived from the template filename (sanitized to alphanumeric), e.g.
> `ANTS8-0Years3T_brain_bias_corrected.nii` → `space-ANTS80Years3Tbrainbiascorrected`

<details>
<summary><strong>📌 For command-line usage of brainprep: click the triangle to expand</strong></summary>

### Command-line usage

**Basic run**
```bash
python brainprep.py \
  --inputs to_preprocess_hc-calgary-preschool.txt \
  --template /path/to/ANTS8-0Years3T_brain_bias_corrected.nii \
  --bids-root /home/andjela/joplin-intra-inter/hc-calgary-preschool \
  --dataset hc-calgary-preschool

**With a template brain mask (skip mask warping)**

```bash
python brainprep.py \
  --inputs to_preprocess_hc-calgary-preschool.txt \
  --template /path/to/template.nii.gz \
  --template-mask /path/to/template_brainmask.nii.gz \
  --bids-root /path/to/bids \
  --dataset hc-calgary-preschool
```

**Skip N4 for selected inputs**

```bash
python brainprep.py \
  --inputs to_preprocess.txt \
  --template /path/to/template.nii.gz \
  --bids-root /path/to/bids \
  --no-bfc no_bfc_list.txt \
  --dataset mydataset
```

**Keep ANTs intermediate files**

```bash
python brainprep.py \
  --inputs to_preprocess.txt \
  --template /path/to/template.nii.gz \
  --bids-root /path/to/bids \
  --keep-work \
  --dataset mydataset
```

**Control parallelism and registration type**

```bash
python brainprep.py \
  --inputs to_preprocess.txt \
  --template /path/to/template.nii.gz \
  --bids-root /path/to/bids \
  --threads 8 \
  --shrink-factor 4 \
  --registration-type a \
  --dataset mydataset
```

</details>

---

## 3) Build the training CSV (`create_dataset_csv.py`)

This step aggregates one or more preprocessed datasets into a **single CSV** used for downstream training (e.g., on Compute Canada). It uses:

* the per-dataset input lists produced earlier (e.g., `preprocess_<dataset>.txt`)
* each dataset’s `participants.tsv` (sex + age for cross-sectional)
* each dataset’s `sessions.tsv` (session-specific age for longitudinal datasets)

It outputs a CSV with (at minimum) these columns:

* `dataset`, `subject_id`, `image_uid`
* `sex` (0 = male, 1 = female, -1 = missing/unknown)
* `age_bef_norm` (raw age in months, rounded to 3 decimals)
* `age` (min–max normalized to [0,1])
* `image_path`, `segm_path`, `latent_path`
* `split` (fold assignment: 1..N)

#### What the script expects

For each dataset, you provide:

* `--bids-roots`: path(s) to BIDS dataset root(s)
* `--layouts`: one per dataset (`long` or `cross`)
* `--input-lists`: one per dataset (`preprocess_<dataset>.txt`)
* `--dest-path-for-images`: where your **training artifacts** are expected to live (brain / segm / latent outputs)

> The script **does not create** brain/segm/latent files — it only writes paths to them in the CSV.

#### Arguments

* `--bids-roots`
  One or more BIDS roots (e.g., `/data/hc-bcp /data/hc-calgary-preschool`)

* `--layouts`
  One per dataset: `cross` or `long`

* `--input-lists`
  One per dataset: text files listing the images to include (one path per line)

* `--age-units` *(optional)*
  `m` = months, `y` = years (converted to months).
  You can provide **one value** (applies to all) or **one per dataset**.

* `--dest-path-for-images`
  Base folder where model inputs are expected to be found:

  * `{dest}/{sub[_ses]}_brain.nii.gz`
  * `{dest}/{sub[_ses]}_segm.nii.gz`
  * `{dest}/{sub[_ses]}_latent.npz`

* `--out-csv` *(optional)*
  Output CSV filename (default: `dataset.csv`)

* `--folds` *(optional)*
  Number of stratified folds (default: 5)

* `--seed` *(optional)*
  Seed for fold assignment (default: 42)

<details>
<summary><strong>📌 Example commands (click the triangle to expand)</strong></summary>

#### Example A — single dataset (longitudinal)

```bash
python create_dataset_csv.py \
  --bids-roots /home/andjela/joplin-intra-inter/hc-calgary-preschool \
  --layouts long \
  --input-lists preprocess_hc-calgary-preschool.txt \
  --age-units y \
  --dest-path-for-images /home/andjela/joplin-intra-inter/hc-calgary-preschool/derivatives/brainprep_export \
  --out-csv hc-calgary-preschool_dataset.csv \
  --folds 5 \
  --seed 42
```

#### Example B — two datasets (both longitudinal)

```bash
python create_dataset_csv.py \
  --bids-roots \
    /home/andjela/joplin-intra-inter/hc-bcp \
    /home/andjela/joplin-intra-inter/hc-calgary-preschool \
  --layouts long long \
  --input-lists \
    preprocess_hc-bcp.txt \
    preprocess_hc-calgary-preschool.txt \
  --age-units y y \
  --dest-path-for-images /scratch/$USER/training_inputs \
  --out-csv combined_dataset.csv
```

#### Example C — mixed layouts (cross-sectional + longitudinal)

```bash
python create_dataset_csv.py \
  --bids-roots \
    /home/andjela/joplin-intra-inter/hc-ping \
    /home/andjela/joplin-intra-inter/hc-calgary-preschool \
  --layouts cross long \
  --input-lists \
    preprocess_hc-ping.txt \
    preprocess_hc-calgary-preschool.txt \
  --age-units m y \
  --dest-path-for-images /scratch/$USER/training_inputs \
  --out-csv mixed_layout_dataset.csv
```

</details>

#### Output CSV example (columns)

You’ll get one row per image (for longitudinal: usually multiple rows per subject):

```text
dataset,subject_id,image_uid,sex,age,age_bef_norm,image_path,segm_path,latent_path,split
hc-calgary-preschool,sub-10001,ses-001,1,0.4123,36.125,"..._brain.nii.gz","..._segm.nii.gz","..._latent.npz",3
...
```

---


Afterwards it is ready for training!
