# CoCo-FL

Coordinated continual federated learning for a digital patient model.

CoCo-FL pre-trains a population backbone over tokenised MIMIC-IV patient
timelines, then adapts it per centre and per patient while keeping the shared
representation frozen. The work is split across work packages:

| WP  | Question                                                          | Entry point          |
| --- | ----------------------------------------------------------------- | -------------------- |
| WP2 | Split one cohort into 3 heterogeneous centres and characterise them | `build_split_v2.py`  |
| WP3 | Does the population T-LSTM generalise on held-out downstream tasks? | `cocofl_train`       |
| WP4 | Does this survive federation across hospital silos (FedAvg)?       | `cocofl_evaluate`    |
| WP5 | Does hierarchical population ↔ cohort ↔ patient coordination help? | `cocofl_evaluate`    |

WP4 and WP5 are two methods inside one harness rather than two runners: one cell
evaluates `population_only`, `local_finetune`, `standard_fedavg`,
`top_down_only` and `hierarchical` against the same split, so the baselines are
comparable by construction.

## Repository layout

```
src/cocofl/
  preprocess/      MEDS → token stream conversion, ICD/ATC vocabulary mapping
  data/            tokenised-timeline datasets, one per downstream task
  models/          T-LSTM + static backbone (TFNFullModel), classifier head
  adapters/        FFN / LoRA-style adapters used for personalisation
  train/           population backbone training loop
  downstream/      head fitting and linear probes on the frozen backbone
  personalisation/ per-patient adaptation + continual-learning ablations
  federated/       hospital splits and the Flower FedAvg round
  hierarchical/    population ↔ cohort ↔ patient coordination
  evaluation/      evaluation harness, baselines, metrics, early stopping
  configs/         Hydra configs (population_training, tokenization, inference, dataset)
scripts/           pipeline launchers, split construction, analysis
scripts/meds/      MIMIC-IV → MEDS extraction (runs before tokenisation)
tests/             unit tests; `pytest tests` needs no data and no GPU
docker/            CUDA 12.4 image; requirements.txt is frozen from the run environment
```

## Requirements

- Python 3.12 (the reported runs used 3.12.3)
- CUDA-capable GPU. One WP4/WP5 cell needs ~6 GiB of VRAM and runs for hours;
  the full 12-cell grid was run on RTX A6000 cards.
- Credentialed access to [MIMIC-IV](https://physionet.org/content/mimiciv/) on
  PhysioNet

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[federated]"
```

`pyproject.toml` pins the versions that matter for reproducibility (torch,
flwr, polars, transformers, pyarrow, protobuf). For a byte-exact environment,
`docker/requirements.txt` is a `pip freeze` of the machine that produced the
reported results.

The shell launchers default `PROJECT` to the repository root and `VENV` to
`$PROJECT/venv`. Override either from the environment if your layout differs.

## Data — read before running

**No patient data is in this repository, by design.** MIMIC-IV is credentialed
under the PhysioNet Data Use Agreement and must not be redistributed. That
applies to the derived artifacts too: the split files are lists of MIMIC
`subject_id`s, so they are regenerated from your own copy rather than shipped
(step 3 below, deterministic under `--seed 42`).

The preprocessing code expects one file that is deliberately **not** committed:

```
src/cocofl/preprocess/maps/mimic-iv_derived.csv.gz
    columns: subject_id, hadm_id, stay_id, first_day_sofa, sepsis
```

It is built from the [MIMIC-IV derived concepts](https://github.com/MIT-LCP/mimic-code)
(`sofa` and `sepsis3`) and is read by
[src/cocofl/preprocess/mappings.py](src/cocofl/preprocess/mappings.py) for
first-day SOFA. Generate it against your own MIMIC-IV instance and drop it at
that path before preprocessing.

The other files in `preprocess/maps/` are public terminologies (ICD-9/10,
ICD-PCS, ATC) and are committed.
[scripts/convert_icds_from_txt_to_csv.py](scripts/convert_icds_from_txt_to_csv.py)
is how they were built.

## Expected directory structure

Code is in the repository; everything below is built locally and ignored by git.

```
coco-fl/
  data/
    mimic_meds/                      scripts/meds/run_mimic.sh writes this
    tokenized/mimic_train/           cocofl_tokenize; must hold static_data.pickle
    federated_splits_v2/             build_split_v2.py            (quantity-only)
    federated_splits_v2_classaware/  build_split_v2.py --class-aware  ← reported
        global_test.json  population_reserve.json  alpha_{0.1,0.5,1.0,10.0}.json
  models/
    population_reserve_nonotes_full/best_model.pt     the backbone
    population_heads/<task>.pt                        one head per task
  results_<tag>_<split>_<task>_a<alpha>/
    config.yaml  wp5_results.csv  predictions_<task>_a<alpha>.npz  patient_records.csv
  venv/                              or set VENV=/path/to/your/env
```

## Reproducing the reported results

`run.sh` is the entry point; `bash run.sh` on its own lists the stages.

```bash
bash run.sh tests      # no data, no GPU
bash run.sh smoke      # whole WP5 path, tiny budget, ~20 min
bash run.sh all        # splits -> backbone -> heads -> grid -> analysis
```

The steps below are what those stages run, in case you want them one at a time.

The reported grid is **12 cells**: the class-aware partition × 3 tasks
(`hospital_mortality`, `readmission`, `icu_admission`) × 4 Dirichlet α
(`0.1, 0.5, 1.0, 10.0`), run end to end through Flower at `seed=42` and
`run_seed=42`. Steps 1–5 build the inputs; step 6 runs the grid; step 7 turns
cells into the reported tables.

Every knob for steps 6–7 has exactly one definition, in
[scripts/wp5_common.sh](scripts/wp5_common.sh). Override values through the
environment; do not fork the list.

```bash
# 1. MIMIC-IV -> MEDS  (see scripts/meds/mimic/README.md)
bash scripts/meds/run_mimic.sh

# 2. MEDS -> token timelines -> data/tokenized/mimic_train
bash scripts/run_tokenization.sh

# 3. splits. Both partitions, same 10/40/50 three-way carve, seed 42.
#    The reported grid uses the CLASS-AWARE one; the quantity-only one is the
#    `random` comparison.
python scripts/build_split_v2.py \
  --tokenized-dir data/tokenized/mimic_train \
  --out-dir data/federated_splits_v2 \
  --alphas 0.1,0.5,1.0,10.0 --num-hospitals 3 --seed 42
python scripts/build_split_v2.py \
  --tokenized-dir data/tokenized/mimic_train \
  --out-dir data/federated_splits_v2_classaware \
  --alphas 0.1,0.5,1.0,10.0 --num-hospitals 3 --seed 42 --class-aware

# 4. population backbone: next-token pre-training on the 40% reserve ONLY.
#    n_train_windows=0 lifts the window cap and num_epochs=5 overrides the
#    config's smoke-scale defaults; together these give the reported backbone,
#    45,889 non-overlapping windows over five epochs.
torchrun --no_python --standalone --nproc_per_node=1 cocofl_train \
  --config-name=population_training \
  data_fp=data/tokenized/mimic_train \
  population_reserve_path=data/federated_splits_v2/population_reserve.json \
  val_size=0.05 num_epochs=5 n_train_windows=0 \
  out_dir=models/population_reserve_nonotes_full \
  device=cuda
# -> best_model.pt (best epoch, the one everything downstream loads),
#    recent_model.pt (last epoch) and config.yaml

# 5. task heads + the population baseline table, from the SAME run.
#    Writes models/population_heads/, which every WP4/WP5 cell then loads.
bash scripts/run_population_baseline.sh

# 6. the reported grid. One cell = one (split, task, alpha); the queue runs
#    them in sequence on one GPU and retries a cell that lost to a transient
#    OOM. `0` = do not wait for another pid first.
GPU=0 bash scripts/v9_queue2.sh 0 0 \
  v11full:classaware:hospital_mortality:0.1:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:hospital_mortality:0.5:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:hospital_mortality:1.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:hospital_mortality:10.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:readmission:0.1:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:readmission:0.5:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:readmission:1.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:readmission:10.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:icu_admission:0.1:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:icu_admission:0.5:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:icu_admission:1.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v11full:classaware:icu_admission:10.0:USE_FLOWER=true,SEED=42,RUN_SEED=42
# -> results_v11full_<split>_<task>_a<alpha>/wp5_results.csv  (one per cell)

# 7. cells -> reported tables
python scripts/summarise_paired.py                    # paired deltas vs population_only
python scripts/analyse_thresholds.py --csv <cell>/wp5_results.csv --out thresholds.md
python scripts/heterogeneity_analysis.py results_v11full_*   # between-centre variance
python scripts/diagnose_patient_streams.py            # patient-evaluability table
```

Two notes on the grid:

- **Cohort budget.** Every cell trains 875 cohort steps (28,000 windows per
  centre per round), the `COHORT_STEPS` default in `wp5_common.sh`.
- **Do not mix Flower and non-Flower cells inside one grid.** Same arithmetic,
  different execution order, so results agree to within ±0.0008 rather than
  exactly. `wp5_common.sh` documents the other knobs that change a *method*
  rather than fix a defect — `PER_HOSPITAL_SELECTION`, `HEAD_BIAS_ONLY`,
  `TRAIN_HEAD`, `LOCAL_FT_PATIENTS` — and a cell run with any of them set is
  not comparable to the reported campaign.

`scripts/run_wp2_pipeline_v2.sh` runs the WP2 deliverable end to end (split,
per-centre characterisation, backbone) on the quantity-only partition with a
10,000-window training cap. That cap and that partition are the WP2
characterisation setting, **not** the inputs the reported WP4/WP5 grid consumes
— for those, follow steps 3–5 above.

Two further console entry points exist and are **not** part of the reported
grid: `cocofl_personalize` runs the per-patient adaptation ablation
(head-only / FFN / continual strategies) standalone, and `cocofl_hierarchical`
runs the hierarchical loop outside the comparison harness. Both are useful for
inspecting one level in isolation; every reported number comes from
`cocofl_evaluate`.

## Seeds, determinism, and how close a re-run gets

Two seeds, deliberately separate, both recorded in every cell's `config.yaml`
and as columns in `wp5_results.csv`:

- `SEED` (42) fixes the **data**: the Dirichlet partition, the balanced-sampling
  draw and the dataloader order. Hold it fixed across a campaign — changing it
  redraws which patients belong to which centre.
- `RUN_SEED` (42) fixes the **training**: torch's global RNG, which initialises
  each adapter. Vary this one to measure run-to-run variance on the same
  partition.

`build_split_v2.py` seeds Python's `random` from `--seed`, so the splits are
exactly reproducible — that is why they are regenerated rather than shipped.

**Expected tolerance.** Re-running a cell at the same two seeds on the same
machine reproduces every reported metric exactly; we verify this by diffing
cell CSVs column by column. Two things move the numbers and are not defects:

- Running the federated steps through Flower rather than the in-process loop
  (`USE_FLOWER`) changes the order of the same arithmetic, so results agree to
  within **±0.0008 AUROC**, not exactly. Do not mix the two inside one grid.
- A different GPU or CUDA version can move the last digits through
  non-deterministic kernels. We do not set
  `torch.use_deterministic_algorithms(True)`, because the cost outweighs the
  benefit at effects of this size — the reported differences vs
  `population_only` are ~0.001–0.003, which is the same order, so the paired
  bootstrap in `scripts/paired_bootstrap.py`, not a point estimate, is what
  carries the conclusions.

**The environment the reported numbers came from**, and the authoritative
reproduction target — development on macOS is validated by re-running
`bash run.sh smoke` here before anything is believed:

| | |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0 |
| Python | 3.12.3 |
| GPU | NVIDIA RTX A6000 (48 GiB), driver 575.57.08, CUDA runtime 12.9 |
| torch | 2.7.1+cu126 (built for CUDA 12.6, cuDNN 9.5.1) |
| federation | flwr 1.33.0 |

`docker/Dockerfile` is built on `nvidia/cuda:12.4.1-devel-ubuntu22.04`, which is
older than the machine above. The torch wheel pinned in
`docker/requirements.txt` ships its own CUDA runtime, so the image works, but it
is not a byte-for-byte image of the reported environment — the table is.

## Validating the pipeline without running a full grid

A full cell is 17 to 24 hours. Two smoke runs exercise the whole path at a tiny
budget and assert that every reported column is actually populated, rather than
merely present in the header:

```bash
GPU=0 bash scripts/smoke_wp5.sh        # all five methods, one cell
GPU=0 bash scripts/smoke_wp4_knobs.sh  # the WP4 ablation knobs
```

They need the same tokenised cohort, splits and checkpoint as a real run. The
unit tests need none of that:

```bash
pytest tests      # 274 passed, 1 skipped — no data, no GPU
```

## Additional analyses

These produce results discussed in the report and are run by hand against
finished cells:

| Script                                                                 | Answers                                                     |
| ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| [paired_bootstrap.py](scripts/paired_bootstrap.py)                     | is one cell's delta vs `population_only` distinguishable from zero |
| [summarise_paired.py](scripts/summarise_paired.py)                     | the same verdict across every testable cell                 |
| [analyse_thresholds.py](scripts/analyse_thresholds.py)                 | is there an α beyond which methods degrade                  |
| [heterogeneity_analysis.py](scripts/heterogeneity_analysis.py)         | between-centre variance against exploitable signal          |
| [subgroup_analysis.py](scripts/subgroup_analysis.py)                   | per-subgroup breakdown of the adapted models                |
| [cross_centre_transfer.py](scripts/cross_centre_transfer.py)           | does a centre's adapter help another centre                 |
| [recalibration_baseline.py](scripts/recalibration_baseline.py)         | does adaptation beat simply recalibrating the population model |
| [diagnose_patient_streams.py](scripts/diagnose_patient_streams.py)     | how often a patient-level acceptance comparison is even defined |
| [wp2_alpha_sweep.py](scripts/wp2_alpha_sweep.py)                       | per-centre class balance and sequence length by α           |

## Note on artifacts

Model checkpoints, run logs, figures and result CSVs are not tracked here. They
are reproduced by the steps above. `notebooks/figures.ipynb` is kept for the
figure definitions it carries, but it imports the upstream `ethos.*` modules
this fork renamed and does not run against this package.

## License

MIT — see [LICENSE](LICENSE).
