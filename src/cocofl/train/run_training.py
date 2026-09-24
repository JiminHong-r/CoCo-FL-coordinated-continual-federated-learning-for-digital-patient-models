import math
import os
import time
from pathlib import Path

import hydra
import torch as th
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ..data.static_features import TFNTimelineDataset, collate_with_static
from ..utils import dump_config, load_model_checkpoint, setup_torch
from .metrics import estimate_loss
from .utils import configure_optimizers, get_lr


@hydra.main(version_base=None, config_path="../configs", config_name="population_training")
def main(cfg: DictConfig):
    """This training script can be run both on a single gpu in debug mode, and also in a larger
    training run with distributed data parallel (ddp).

    To run on a single GPU, example:
    $ cocofl_train [args...]

    To run with DDP on 4 gpus on 1 node, example:

    $ torchrun --standalone --nproc_per_node=4 cocofl_train [args...]

    To run with DDP on 4 gpus across 2 nodes, example:

    - Run on the first (master) node with example IP 123.456.123.456:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456
     --master_port=1234 cocofl_train [args...]

    - Run on the worker node:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456
     --master_port=1234 cocofl_train [args...]

    (If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
    """
    device = cfg.device
    out_dir = Path(cfg.out_dir)
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=cfg.backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        th.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert cfg.gradient_accumulation_steps % ddp_world_size == 0
        cfg.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0

    tokens_per_iter = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.n_positions

    if master_process:
        logger.info(f"Tokens per iteration per worker: {tokens_per_iter:,}")
        out_dir.mkdir(parents=True, exist_ok=True)
        # The resolved config beside the checkpoint it produced. Which
        # `num_epochs` and `n_train_windows` a backbone was trained with is not
        # recoverable from the .pt, and both change the model substantially.
        dump_config(cfg, out_dir / "config.yaml")
    ctx = setup_torch(device, cfg.dtype, 42 + seed_offset)

    train_dataset = TFNTimelineDataset(cfg.data_fp, n_positions=cfg.n_positions)
    vocab = train_dataset.vocab

    vocab_size = math.ceil(len(vocab) / 64) * 64

    # WP2 task 3: restrict population pre-training to the EXCLUSIVE 40%
    # population reserve. When ``population_reserve_path`` is set, keep only
    # windows whose patient_id is in that reserve, then split those into
    # train/val. This guarantees the backbone never sees a centre's patients
    # or the global held-out test set.
    reserve_path = cfg.get("population_reserve_path", None)
    if reserve_path:
        import json
        reserve = {int(p) for p in
                   json.loads(Path(reserve_path).read_text())["patient_ids"]}
        full_ds = train_dataset
        n_windows = len(full_ds)
        # Build the per-window patient-id array vectorised. ``patient_id_at_idx``
        # is a lazy sharded accessor (scalar indexing only), so instead we use
        # the fact that token positions [offset_i, offset_{i+1}) all belong to
        # patient i: repeat each patient id over its token span.
        pids_per_patient = full_ds.patient_ids
        offsets = full_ds.patient_offsets
        total_tokens = len(full_ds.tokens)
        counts = th.diff(th.cat([offsets, offsets.new_tensor([total_tokens])]))
        pid_per_token = th.repeat_interleave(pids_per_patient, counts)
        pid_at_idx = pid_per_token[:n_windows]
        mask = th.isin(pid_at_idx, th.tensor(sorted(reserve), dtype=pid_at_idx.dtype))
        keep = th.nonzero(mask, as_tuple=False).flatten()
        if keep.numel() == 0:
            raise ValueError(
                f"population_reserve_path={reserve_path} matched 0 windows in "
                f"{cfg.data_fp}; reserve patient IDs are not in this pool."
            )
        perm = th.randperm(keep.numel(), generator=th.Generator().manual_seed(42))
        keep = keep[perm]
        n_val = max(1, int(keep.numel() * cfg.val_size))
        # Keep indices as tensors (matches train_test_split's th.arange pattern)
        # to avoid materialising a ~100M-element Python list.
        val_dataset = Subset(full_ds, keep[:n_val])
        train_dataset = Subset(full_ds, keep[n_val:])
        if master_process:
            logger.info(
                f"population_reserve filter: kept {keep.numel():,}/{n_windows:,} "
                f"windows ({len(reserve):,} reserve patients) -> "
                f"train={len(train_dataset):,} val={len(val_dataset):,}"
            )
    else:
        train_dataset, val_dataset = train_dataset.train_test_split(cfg.val_size)

    # TimelineDataset returns one sample PER TOKEN OFFSET (sliding window),
    # so naive len() is ~total_tokens. For TFN-style epochs we instead iterate
    # over non-overlapping windows of size n_positions.
    epoch_stride = cfg.get("epoch_stride", None) or cfg.n_positions
    if epoch_stride > 1:
        train_dataset = Subset(train_dataset, list(range(0, len(train_dataset), epoch_stride)))
        val_dataset = Subset(val_dataset, list(range(0, len(val_dataset), epoch_stride)))

    # Optional cap on number of training windows (TFN-style: ~10K patients ~= a few thousand windows)
    n_train_windows = cfg.get("n_train_windows", None)
    if n_train_windows and n_train_windows > 0 and len(train_dataset) > n_train_windows:
        import random as _random
        _random.seed(42)
        idxs = sorted(_random.sample(range(len(train_dataset)), int(n_train_windows)))
        train_dataset = Subset(train_dataset, idxs)
        if master_process:
            logger.info(f"Capped train windows to {len(train_dataset):,}")

    collate_fn = collate_with_static

    train_dataloader, val_dataloader = (
        DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=not ddp,
            sampler=DistributedSampler(dataset) if ddp else None,
            collate_fn=collate_fn,
        )
        for dataset in [train_dataset, val_dataset]
    )
    # A proper epoch loop, matching the TFN training notebook style.

    num_epochs = cfg.get("num_epochs", None)
    if num_epochs is None or num_epochs <= 0:
        raise ValueError(
            "num_epochs must be set (TFN-style training). "
            "Iteration-based training has been removed; use num_epochs=N instead."
        )

    eval_iters = len(val_dataset) // (cfg.batch_size * cfg.n_positions) + 1

    if master_process:
        logger.info(
            "Train dataset size: {:,}, Val dataset size: {:,} "
            "(num_epochs={}, eval_iters={}, batches_per_epoch~={:,})".format(
                len(train_dataset),
                len(val_dataset),
                num_epochs,
                eval_iters,
                len(train_dataset) // cfg.batch_size,
            )
        )

    def _to_device(x, y):
        y = y.to(device, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return tuple(t.to(device, non_blocking=True) for t in x), y
        return x.to(device, non_blocking=True), y

    iter_num, best_val_loss, best_metric_score, optimizer_state, wandb_path = 0, 1e9, 0, None, None
    if cfg.resume:
        model_fp = out_dir / "recent_model.pt"
        logger.info(f"Resuming from the most recent model: {model_fp}")
        raw_model, checkpoint = load_model_checkpoint(model_fp, map_location=device)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        best_metric_score = checkpoint["best_metric_score"]
        optimizer_state = checkpoint["optimizer"]
        wandb_path = checkpoint["wandb_path"]
    else:
        from ..models.backbone import TFNFullModel
        from ..data.static_features import CATEGORICAL_CARDINALITIES, NUMERICAL_DIM
        raw_model = TFNFullModel(
            vocab_size=vocab_size,
            n_embd=cfg.n_embd,
            n_layer=cfg.n_layer,
            n_positions=cfg.n_positions,
            categorical_cardinalities=CATEGORICAL_CARDINALITIES,
            numerical_dim=NUMERICAL_DIM,
            use_static=True,
        )

        if master_process:
            logger.info(f"Initializing a new model from scratch: {raw_model.config}")
    logger.info(f"Model parameters: {raw_model.num_parameters() / 1e6:.2f}M")

    raw_model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = th.amp.GradScaler(enabled=(cfg.dtype == "float16"))
    # optimizer
    optimizer = configure_optimizers(
        raw_model, cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    num_params = raw_model.num_parameters()
    if master_process:
        logger.info(f"Number of parameters: {num_params / 1e6:.2f}M")
        logger.info(("Not c" if cfg.no_compile else "C") + "ompiling the model...")
    model = th.compile(raw_model, disable=cfg.no_compile)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # logging
    online_logger, wandb_run = None, None
    if cfg.wandb_log and master_process:
        import wandb

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        dataset_name = Path(cfg.data_fp).parts[-2]
        cfg_dict.update(
            {
                "dataset": dataset_name,
                "vocab_size": len(vocab),
                "vocab_size_train": vocab_size,
                "model_num_params": num_params,
                "model_num_params_total": raw_model.num_parameters(exclude_embeddings=False),
            }
        )
        run_id = wandb_path.split("/")[-1] if wandb_path is not None else None
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg_dict,
            tags=[dataset_name],
            resume_from=f"{run_id}?_step={iter_num}" if run_id is not None else None,
        )
        online_logger = wandb

    # ---- TFN-style epoch-based training loop ----
    iter_num = 0
    local_iter_num = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        if master_process:
            logger.info(f"=== Starting epoch {epoch + 1}/{num_epochs} ===")
        if ddp and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for batch_idx, (X_cpu, Y_cpu) in enumerate(train_dataloader):
            X, Y = _to_device(X_cpu, Y_cpu)

            # Learning rate schedule (TFN uses constant lr, so allow disabling)
            if cfg.get("use_lr_schedule", True):
                lr = get_lr(iter_num, cfg)
            else:
                lr = cfg.lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # forward-backward (with optional gradient accumulation)
            for micro_step in range(cfg.gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == cfg.gradient_accumulation_steps - 1
                    )
                with ctx:
                    # X is (tokens, elapsed_times, static_cat, static_num)
                    output = model(
                        input_ids=X[0], elapsed_times=X[1],
                        static_categorical=X[2], static_numerical=X[3], labels=Y,
                    )
                    loss = output.loss / cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()

            if cfg.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                th.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % cfg.log_interval == 0 and master_process:
                lossf = loss.item() * cfg.gradient_accumulation_steps
                logger.info(
                    f"[ep{epoch + 1} b{batch_idx} it{iter_num}]: "
                    f"loss={lossf:.4f}, time={dt * 1000:.0f}ms"
                )

            iter_num += 1
            local_iter_num += 1

        # End-of-epoch: evaluate + checkpoint (TFN style)
        th.cuda.empty_cache()
        losses = estimate_loss(
            model,
            ctx,
            loaders=[("val", val_dataloader)],
            eval_iters=eval_iters,
        )
        if ddp:
            for key in list(losses.keys()):
                gathered = [None] * ddp_world_size
                th.distributed.all_gather_object(gathered, losses[key])
                losses[key] = sum(gathered) / ddp_world_size

        if master_process:
            logger.info(
                "epoch {}/{} done: val loss {:.4f}".format(
                    epoch + 1, num_epochs, losses["loss/val"]
                )
            )
            checkpoint = {
                "iter_num": iter_num,
                "epoch": epoch + 1,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": losses["loss/val"],
                "best_metric_score": best_metric_score,
                "model_config": raw_model.config,
                "vocab": vocab.stoi,
                "model_type": "population",
                "wandb_path": wandb_run.path if wandb_run is not None else None,
            }
            th.save(checkpoint, out_dir / "recent_model.pt")
            logger.info("Saved the most recent model.")
            if losses["loss/val"] < best_val_loss:
                th.save(checkpoint, out_dir / "best_model.pt")
                logger.info(f"Saved the best model: {best_val_loss} => {losses['loss/val']}")
                best_val_loss = losses["loss/val"]
            if online_logger is not None:
                online_logger.log(
                    {
                        "other/epoch": epoch + 1,
                        "other/iter": iter_num,
                        "other/lr": lr,
                        **losses,
                    }
                )

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
