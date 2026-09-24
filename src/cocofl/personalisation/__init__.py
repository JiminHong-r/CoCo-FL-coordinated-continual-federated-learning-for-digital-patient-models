"""Per-patient personalisation of the population backbone.

Three strategies (all attach a fresh adapter per patient on top of the frozen
backbone, train on the patient's early data, evaluate on their late data):

- :func:`personalize_head_only`  : only the classifier MLP head trains.
- :func:`personalize_ffn`        : Houlsby FFN adapter + head train.
- :func:`personalize_full`       : full fine-tune (catastrophic forgetting reference).

Continual-learning regularisers (EWC / distillation / replay) live in
:mod:`cocofl.personalisation.continual` and combine with any strategy via
the ``extra_loss_fn`` hook.
"""
