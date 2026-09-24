from contextlib import nullcontext
from pathlib import Path

import torch as th

from .models.backbone import TFNFullModel


def setup_torch(device, dtype, seed=42):
    th.manual_seed(seed)
    device_type = "cuda" if "cuda" in device else "cpu"
    if dtype == "bfloat16" and device_type == "cuda" and not th.cuda.is_bf16_supported():
        print("WARNING: bfloat16 is not supported on this device, using float16 instead")
        dtype = "float16"
    if device_type == "cuda":
        th.cuda.manual_seed(seed)
        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True
    ptdtype = {"float32": th.float32, "bfloat16": th.bfloat16, "float16": th.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else th.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    return ctx


def load_model_checkpoint(checkpoint_fp: str | Path, **kwargs) -> tuple[TFNFullModel, dict]:
    """Rebuild the population backbone from a checkpoint; ``**kwargs`` go to ``torch.load``.

    Rebuilds a ``TFNFullModel`` the same way ``build_population_classifier`` does.
    """
    checkpoint = th.load(checkpoint_fp, weights_only=False, **kwargs)
    cfg = checkpoint["model_config"]

    model = TFNFullModel(
        vocab_size=cfg.vocab_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_positions=cfg.n_positions,
        categorical_cardinalities=tuple(cfg.categorical_cardinalities),
        numerical_dim=cfg.numerical_dim,
        use_static=cfg.use_static,
        num_heads=getattr(cfg, "num_heads", 2),
    )
    model.load_state_dict(checkpoint["model"])

    del checkpoint["model"]
    del checkpoint["model_config"]
    return model, checkpoint


def dump_config(cfg, path) -> None:
    """Write the resolved config next to the results, without ever failing the run.

    ``resolve=True`` fills in interpolations, so the file records the real paths
    rather than ``${...}``. If resolving fails (e.g. a custom resolver), an
    unresolved dump is written instead; a long run should not stop over its record
    file.
    """
    from omegaconf import OmegaConf

    try:
        Path(path).write_text(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        Path(path).write_text(
            f"# resolve failed: {exc!r}\n" + OmegaConf.to_yaml(cfg, resolve=False)
        )
