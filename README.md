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

## Reproducing the reported results

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
#    No window cap — the reported backbone saw 45,889 non-overlapping windows.
torchrun --no_python --standalone --nproc_per_node=1 cocofl_train \
  --config-name=population_training \
  data_fp=data/tokenized/mimic_train \
  population_reserve_path=data/federated_splits_v2/population_reserve.json \
  val_size=0.05 \
  out_dir=models/population_reserve_nonotes_full \
  device=cuda

# 5. task heads + the population baseline table, from the SAME run.
#    Writes models/population_heads/, which every WP4/WP5 cell then loads.
bash scripts/run_population_baseline.sh

# 6. the reported grid. One cell = one (split, task, alpha); the queue runs
#    them in sequence on one GPU and retries a cell that lost to a transient
#    OOM. `0` = do not wait for another pid first.
GPU=0 bash scripts/v9_queue2.sh 0 0 \
  v10flw:classaware:hospital_mortality:0.1:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:hospital_mortality:0.5:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:hospital_mortality:1.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:hospital_mortality:10.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:readmission:0.1:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:readmission:0.5:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:readmission:1.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:readmission:10.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:icu_admission:0.1:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:icu_admission:0.5:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:icu_admission:1.0:USE_FLOWER=true,SEED=42,RUN_SEED=42 \
  v10flw:classaware:icu_admission:10.0:USE_FLOWER=true,SEED=42,RUN_SEED=42
# -> results_v10flw_<split>_<task>_a<alpha>/wp5_results.csv  (one per cell)

# 7. cells -> reported tables
python scripts/summarise_paired.py                    # paired deltas vs population_only
python scripts/analyse_thresholds.py --csv <cell>/wp5_results.csv --out thresholds.md
python scripts/heterogeneity_analysis.py results_v10flw_*   # between-centre variance
python scripts/diagnose_patient_streams.py            # patient-evaluability table
```

Two notes on the grid:

- **The robustness grid** re-runs the same 12 cells at 8.75× the cohort budget
  (`COHORT_STEPS=875`, i.e. 28,000 windows per centre per round instead of
  3,200). Pass it through the queue's per-cell env field and give it its own
  tag, so the two campaigns land in separate directories.
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

## Validating the pipeline without running a full grid

A full cell is about 10 hours. Two smoke runs exercise the whole path at a tiny
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
