"""End-to-end wiring of the harness settings that control sample size.

Runs the real ``run_hierarchical_bidir`` and baselines on a tiny TFN backbone, to
check the separate trajectory loaders, explicit patient pooling and recorded
eval-set sizes.
"""
from __future__ import annotations

import torch

from cocofl.evaluation.baselines import eval_population_only, run_local_finetune
from cocofl.evaluation.batching import make_eval_loader
from cocofl.evaluation.run_evaluation import _cap, run_hierarchical_bidir
from cocofl.hierarchical.cohort import CohortConfig
from cocofl.hierarchical.coordinator import CoordinatorConfig
from cocofl.hierarchical.patient import PatientConfig
from cocofl.hierarchical.run_hierarchical import (
    build_hospital_train_loader, build_patient_streams,
)
from cocofl.models.backbone import TFNFullModel
from cocofl.models.classifier import TFNDownstreamClassifier
from cocofl.personalisation import loop as loop_mod

from test_eval_batching import FakeInferenceDataset, collate, label_fn


def _classifier():
    torch.manual_seed(0)
    # numerical_dim=5 matches the shared `collate` fixture's static_numerical.
    backbone = TFNFullModel(vocab_size=200, n_embd=16, n_layer=1, n_positions=64,
                            categorical_cardinalities=(2, 6, 5), numerical_dim=5)
    return TFNDownstreamClassifier(backbone, num_classes=1)


def _dataset():
    loop_mod._PATIENT_BUCKET_CACHE.clear()
    ds = FakeInferenceDataset()
    # One patient with plenty of windows so patient streams are non-empty.
    ds.start_indices = torch.cat([torch.arange(1, 25), torch.arange(31, 55)])
    ds.patient_id_at_idx = torch.tensor([0] * 30 + [1] * 30)
    return ds


def _coord_cfg(**kw):
    return CoordinatorConfig(
        cohort_cfg=CohortConfig(bottleneck=4, lr=1e-3, local_steps_per_epoch=2),
        patient_cfg=PatientConfig(bottleneck=4, lr=5e-3, n_iters=2),
        patients_per_round=2,
        population_update_freq=2,
        **kw,
    )


class CountingLoader:
    """Wraps a DataLoader and counts how many windows are pulled from it."""

    def __init__(self, loader):
        self.loader = loader
        self.windows = 0

    def __iter__(self):
        for batch in self.loader:
            if batch is not None:
                self.windows += batch[0][0].size(0)
            yield batch

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


def _loaders(ds, n_test, n_traj):
    idxs = list(range(len(ds)))
    return (
        {0: make_eval_loader(ds, _cap(idxs, n_test), collate, batch_size=4)},
        {0: make_eval_loader(ds, _cap(idxs, n_traj), collate, batch_size=4)},
        {0: build_hospital_train_loader(ds, idxs[:8], collate)},
    )


def test_trajectory_loader_absorbs_the_per_round_passes():
    """The test set is read once; the per-round passes go to the trajectory loader, so a full-size test set stays affordable."""
    ds = _dataset()
    device = torch.device("cpu")
    test_lds, traj_lds, train_lds = _loaders(ds, n_test=0, n_traj=4)
    counted_test = {0: CountingLoader(test_lds[0])}
    counted_traj = {0: CountingLoader(traj_lds[0])}
    streams = {0: build_patient_streams(ds, [0, 1], collate, min_samples=4,
                                        max_patients=2, eval_batch_size=4)}
    assert streams[0], "fixture must produce patient streams"

    res = run_hierarchical_bidir(
        _classifier(), train_lds, streams, label_fn, device, _coord_cfg(),
        eval_loaders=counted_test, val_loaders=counted_traj,
        traj_loaders=counted_traj,
        max_rounds=3, patience=999, min_rounds=3,
    )
    assert res.n_rounds == 3
    # Test set: exactly one pass, at the end.
    assert counted_test[0].windows == len(ds)
    # Trajectory set: 2 passes per round (before and after patient feedback) + 1
    # validation pass, plus one round-0 validation pass. Round 0 (identity adapter =
    # the population model) is scored so it can be selected; see
    # tests/test_model_selection.py.
    assert counted_traj[0].windows == (3 * 3 + 1) * 4


def test_traj_defaults_to_the_test_loader():
    """Without traj_loaders, the test loader is used for the trajectory too."""
    ds = _dataset()
    device = torch.device("cpu")
    test_lds, _, train_lds = _loaders(ds, n_test=4, n_traj=4)
    counted = {0: CountingLoader(test_lds[0])}
    streams = {0: build_patient_streams(ds, [0, 1], collate, min_samples=4,
                                        max_patients=2, eval_batch_size=4)}
    run_hierarchical_bidir(
        _classifier(), train_lds, streams, label_fn, device, _coord_cfg(),
        eval_loaders=counted, val_loaders=counted,
        max_rounds=2, patience=999, min_rounds=2,
    )
    # 1 round-0 validation pass + 2 rounds x (2 trajectory + 1 validation) + 1 final
    # pass = 8 passes over 4 windows. That multiple is why a full test set needs a
    # separate trajectory loader.
    assert counted[0].windows == 8 * 4


def test_patient_eval_n_is_explicit_and_overrides_the_n_iters_heuristic():
    ds = _dataset()
    device = torch.device("cpu")
    test_lds, traj_lds, train_lds = _loaders(ds, n_test=8, n_traj=4)
    streams = {0: build_patient_streams(ds, [0, 1], collate, min_samples=4,
                                        max_patients=2, eval_batch_size=4)}
    # Streams are (patient_id, fit, val, eval): acceptance is decided on val so eval
    # stays held out, and the id is kept for the axis 2 subgroup analysis.
    counted = [CountingLoader(e) for _, _, _, e in streams[0]]
    streams = {0: [(pid, t, v, c) for (pid, t, v, _), c in zip(streams[0], counted)]}

    run_hierarchical_bidir(
        _classifier(), train_lds, streams, label_fn, device, _coord_cfg(),
        eval_loaders=test_lds, val_loaders=traj_lds, traj_loaders=traj_lds,
        patient_eval_n=2, max_rounds=1, patience=999, min_rounds=1,
    )
    # Both patient streams are pooled.
    assert all(c.windows > 0 for c in counted)


def test_population_only_records_eval_set_size():
    ds = _dataset()
    test_lds, _, _ = _loaders(ds, n_test=0, n_traj=4)
    res = eval_population_only(_classifier(), test_lds, label_fn,
                               torch.device("cpu"))
    assert res.ns[0] == len(ds)
    assert 0 < res.n_pos[0] < len(ds)


def test_uncapped_eval_scores_every_window():
    ds = _dataset()
    device = torch.device("cpu")
    test_lds, traj_lds, train_lds = _loaders(ds, n_test=0, n_traj=4)
    res = run_local_finetune(
        _classifier(), train_lds, traj_lds, test_lds, label_fn, device,
        _coord_cfg(), max_rounds=2, patience=999, min_rounds=2,
    )
    assert res.aucs[0] is not None
    pop = eval_population_only(_classifier(), test_lds, label_fn, device)
    assert pop.ns[0] == len(ds) > 4  # genuinely larger than the capped set


def test_use_flower_refuses_to_fall_back_when_flwr_is_missing():
    """If Flower is requested but not installed, the harness must raise, not quietly run the in-process path.

    Every setting is read with a default, so a missing setting looks like one left
    off on purpose; a warning would let a run be credited to the wrong runtime.
    """
    import inspect

    from cocofl.evaluation import run_evaluation

    src = inspect.getsource(run_evaluation.main)
    i = src.index("use_flower = bool(")
    guard = src[i:i + 1500]
    assert "import flwr" in guard, "the flwr import must be attempted eagerly"
    assert "raise RuntimeError" in guard, (
        "use_flower=true with flwr unavailable must raise — falling back "
        "attributes the result to the wrong pipeline"
    )
    assert "federated runtime:" in guard, (
        "the resolved runtime must be printed, so a log says which pipeline ran"
    )
