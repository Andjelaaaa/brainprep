
# brainprep 🧠
Preprocessing for pediatric (1–7 yo) MRI brain data

This preprocessing follows the BIDS standard and writes all outputs directly into your dataset’s `derivatives/brainprep/` folder. It works with any BIDS-compliant dataset that provides an `exclude.yaml` file. After raw-image QC, the workflow runs in three steps and supports both cross-sectional and longitudinal datasets.


Pipeline (per T1w): 

**SynthStrip → N4 → affine registration → SynthSeg → WhiteStripe intensity normalization**.

> Full usage + details (exclude rules, layouts, age filters, outputs) live in the documentation.

---

## Installation (conda)

```bash
conda env create -f environment.yml
conda activate brainprep
````

Quick check:

```bash
python brainprep.py --help
```

---

## Workflow

### 1) Create the input list (`create_input_txt.py`)

Build a `.txt` list of absolute paths to the images you want to preprocess.

```bash
python create_input_txt.py /path/to/bids_root -l long --modality T1w -o to_preprocess.txt
```

---

### 2) Run preprocessing (`brainprep.py`)

Run the pipeline and write outputs into `derivatives/brainprep/`.

```bash
python brainprep.py \
  --inputs to_preprocess.txt \
  --template /path/to/template.nii.gz \
  --bids-root /path/to/bids_root \
  --dataset mydataset
```

---

### 3) Build the training CSV (`create_dataset_csv.py`)

Aggregate one or more datasets into a single CSV (paths + demographics + folds).

```bash
python create_dataset_csv.py \
  --bids-roots /path/to/hc-calgary-preschool \
  --layouts long \
  --input-lists preprocess_hc-calgary-preschool.txt \
  --age-units y \
  --dest-path-for-images /scratch/$USER/training_inputs \
  --out-csv dataset.csv
```

---

## Project structure

```
brainprep/
├── brainprep.py
├── create_input_txt.py
├── create_dataset_csv.py
├── environment.yml
```

## Documentation

## Authors

Andjela Dimitrijevic

## How to cite

Publication coming soon...
