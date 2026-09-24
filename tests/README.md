# What the test suite is for

282 tests across 30 files. That is a lot for a project this size, and it is
deliberate: this work's main correctness finding is that **31 silent defects**
produced publishable-looking numbers (RESULTS.md §Correctness). Each defect that
could recur is pinned by a test, so the suite *is* part of the contribution rather
than scaffolding around it. §4.6 of the report cites it.

Nothing here is decorative — every file guards a specific documented failure. Use
this map to find the one you need; you should rarely have to read the others.

## The four that close *classes* of failure, not instances

These are the ones worth naming in the report. They catch categories of bug that
ordinary unit tests pass straight over.

| file | what it caught |
|---|---|
| `test_call_contract.py` | every keyword at every call site vs the callee's real signature — **197 tests passed while the harness called a function with a parameter it did not have** |
| `test_no_undefined_names.py` | any function referencing a name it never binds — **240 tests passed with a latent `NameError` in the FedAvg path** |
| `test_selection_reporting.py` | results rows that contradict the model they describe (D25/D26, which arose in *two independent copies at once*) |
| `test_seeding.py` | data seed vs training seed kept separable, so a replicate cannot silently redraw the patient partition (D29) |

## By area

**Adapters and the level hierarchy**
- `test_adapter_behaviour.py` — identity at init (so "round 0 *is* the population
  model" holds exactly), adapters in the forward path, exactly two slots (D19)
- `test_head_travels.py` — what a level trains is what it persists and is scored
  with (D8)
- `test_refresh_ordering.py` — the top-down refresh must not alter the state just
  evaluated (D14, the 27 % attenuation)

**Patient layer**
- `test_patient_evaluability.py` — "rejected" must not be conflated with
  "unjudgeable" (D20; 0/40 evaluable for mortality)
- `test_delta_patient_pairing.py` — Δ_patient on identical windows (D17, the
  spurious +0.093)
- `test_patient_selection.py`, `test_patient_streams.py`, `test_patient_rotation.py`

**WP4 spec compliance**
- `test_wp4_weighting.py` — knobs reach the coordinator; `recency` decays by
  elapsed time, not list position (D22)
- `test_wp4_triggering.py` — the divergence trigger can actually fire (D10, D23)
- `test_wp4_spec_compliance.py`, `test_wp4_class_weighting.py`,
  `test_adaptation_improvements.py`

**WP5 reporting**
- `test_macro_evidence_floor.py` — MACRO means the same thing at every α (D21,
  D30)
- `test_model_selection.py` — report the best validation round, not the last (D5)
- `test_val_sizing.py` — validation sized by positives, not windows
- `test_wp5_axis_coverage.py`, `test_wp5_eval_wiring.py`

**Data integrity**
- `test_hospital_split_binding.py` — patient lookups keyed by token index (D1:
  without it α controlled nothing)
- `test_eval_no_leakage.py` — never score a method on windows it trained on
- `test_population_probe.py` — the probe survives class imbalance (D3)

**Batching** — `test_eval_batching.py`, `test_train_batching.py`: length-bucketed
batching must not change a single prediction, which is what makes eval batch size
free to vary for memory.

**Statistics and federation**
- `test_paired_bootstrap.py` — one resample applied to both methods
- `test_flower_equivalence.py` — Flower's `FedAvg` equals the in-tree aggregator,
  so "a real FL runtime changes no reported number" is verified not asserted
- `test_baseline_distinction.py` — `local_finetune` and `top_down_only` really are
  different methods (D18)

## Running them

```
python -m pytest tests/ -q            # all 282, ~9s
python -m pytest tests/ -q -k patient # one area
```

The suite needs no GPU and no data: everything is synthetic or static analysis.
`test_flower_equivalence.py` skips automatically if `flwr` is not installed
(`pip install -e '.[federated]'`).

## If you want fewer files

The 30 files could merge into ~9 by area without losing a single test. That is
cosmetic — pytest does not care, and the grouping above already gives you the
navigation. Worth doing only if the file count itself is a problem.
