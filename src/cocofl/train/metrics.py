import math
import os

import torch as th
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


@th.inference_mode()
def estimate_loss(
    model: PreTrainedModel, ctx, loaders: list[tuple[str, DataLoader]], eval_iters: int
) -> dict:
    rank = int(os.environ.get("RANK", -1))
    is_distributed = rank != -1

    if is_distributed:
        eval_iters = math.ceil(eval_iters / int(os.environ["WORLD_SIZE"]))

    device = next(model.parameters()).device
    out = {}
    for split, dataloader in loaders:
        losses = th.empty(eval_iters, device=device)
        for i, (X, Y) in zip(range(eval_iters), dataloader):
            Y = Y.to(device, non_blocking=True)
            if isinstance(X, (tuple, list)):
                X = tuple(t.to(device, non_blocking=True) for t in X)
            else:
                X = X.to(device, non_blocking=True)
            with ctx:
                if isinstance(X, tuple) and len(X) == 4:
                    # TFN_FULL: (tokens, elapsed, static_cat, static_num)
                    output = model(
                        input_ids=X[0], elapsed_times=X[1],
                        static_categorical=X[2], static_numerical=X[3], labels=Y,
                    )
                elif isinstance(X, tuple):
                    output = model(input_ids=X[0], decoder_input_ids=X[1], labels=Y)
                else:
                    output = model(input_ids=X, labels=Y)
                loss = output.loss
            losses[i] = loss.item()

        out[f"loss/{split}"] = losses.mean().item()
    return out
