# CoCo-FL

Coordinated continual federated learning for a digital patient model.

CoCo-FL trains a population-level transformer backbone over tokenised MIMIC-IV
patient timelines, then personalises it per patient while keeping the shared
representation intact. The work is split across work packages:

| WP  | Question                                                     | Entry point           |
| --- | ------------------------------------------------------------ | --------------------- |
| WP2 | Can a population backbone beat a T-LSTM baseline downstream?  | `cocofl_train`        |
| WP3 | How much personalisation is possible without forgetting?      | `cocofl_personalize`  |
| WP4 | Does this survive federation across hospital silos (FedAvg)?  | `cocofl_evaluate`     |
| WP5 | Does hierarchical population ↔ cohort ↔ patient coordination help? | `cocofl_evaluate`     |

WP4 and WP5 are two methods inside one harness rather than two runners: a cell
evaluates `population_only`, `local_finetune`, `standard_fedavg`, `top_down_only`
and `hierarchical` against the same split, so the baselines are comparable by
construction.

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
scripts/           SLURM / shell launchers and analysis utilities
docker/            CUDA 12.4 image used for cluster runs
```

## Requirements

- Python ≥ 3.12
- CUDA-capable GPU for training (see [docker/Dockerfile](docker/Dockerfile))
- Credentialed access to [MIMIC-IV](https://physionet.org/content/mimiciv/) on PhysioNet

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[federated]"
```

## Data — read before running

**No patient data is contained in this repository, by design.** MIMIC-IV is
credentialed under the PhysioNet Data Use Agreement and must not be
redistributed. You need to obtain it yourself and build the derived files
locally.

The preprocessing code expects one file that is deliberately **not** committed:

```
src/cocofl/preprocess/maps/mimic-iv_derived.csv.gz
    columns: subject_id, hadm_id, stay_id, first_day_sofa, sepsis
```

It is built from the [MIMIC-IV derived concepts](https://github.com/MIT-LCP/mimic-code)
(`sofa` and `sepsis3`) and is read by
[src/cocofl/preprocess/mappings.py](src/cocofl/preprocess/mappings.py) for
first-day SOFA. Generate it against your own MIMIC-IV instance and drop it in
that path before preprocessing.

The other files in `preprocess/maps/` are public terminologies (ICD-9/10, ICD-PCS,
ATC) and are committed.

## Pipeline

```bash
# 1. tokenise MEDS-format MIMIC-IV into patient timelines
scripts/run_tokenization.sh

# 2. WP2: build the 10/40/50 split, train the population backbone on the
#    40% reserve, fit the heads and sweep alpha
scripts/run_wp2_pipeline_v2.sh

# 3. federated + hierarchical evaluation (WP4 / WP5)
#    one cell = one (split, task, alpha); the queue runs them in sequence
GPU=1 bash scripts/v9_queue2.sh 1 0 \
  v10:classaware:hospital_mortality:0.5:USE_FLOWER=true,SEED=42,RUN_SEED=42
```

Console entry points are declared in [pyproject.toml](pyproject.toml) if you
prefer to invoke stages directly (`cocofl_tokenize`, `cocofl_train`,
`cocofl_finetune`, `cocofl_personalize`, `cocofl_hierarchical`,
`cocofl_evaluate`).

Scripts under `scripts/` carry SLURM headers for cluster submission; they can
also be run as plain shell scripts from the project root.

## Note on artifacts

Model checkpoints, run logs, figures, and result CSVs are not tracked here —
they are reproduced by the pipeline above and shared separately.

## License

MIT — see [LICENSE](LICENSE).
