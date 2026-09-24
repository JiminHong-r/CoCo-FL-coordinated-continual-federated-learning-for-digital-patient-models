"""Do the adapters behave as the design assumes?

1. Identity at initialisation: "round 0 is the population model" is only true if a
freshly attached adapter changes nothing. 2. In the forward path: a trained adapter
must change predictions, or every method would silently equal population_only. 3.
Cohort and patient compose: patient models start from the cohort model, so both
adapters apply, cohort first. 4. Exactly two slots: cohort and patient. ``attr`` is
required and keyword-only, so a caller cannot attach a third adapter by forgetting
to name the level.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from cocofl.adapters.ffn import FFNAdapter, attach_ffn_adapter


H, B, T = 16, 3, 7


class _Backbone(nn.Module):
    """Mimics TFNFullModel's adapter contract: two slots, cohort then patient."""

    def __init__(self):
        super().__init__()
        self.n_embd = H
        self.encoder = nn.Linear(H, H)
        self.cohort_adapter = None
        self.patient_adapter = None

    def forward(self, x):
        h = self.encoder(x)
        if self.cohort_adapter is not None:
            h = self.cohort_adapter(h)
        if self.patient_adapter is not None:
            h = self.patient_adapter(h)
        return h


def _x():
    torch.manual_seed(0)
    return torch.randn(B, T, H)


# 1. Identity at initialisation

def test_fresh_adapter_is_exactly_the_identity():
    a = FFNAdapter(H, bottleneck=4)
    x = _x()
    assert torch.equal(a(x), x), "round 0 must BE the population model, not merely close"


def test_identity_holds_because_the_up_projection_is_zeroed():
    a = FFNAdapter(H, bottleneck=4)
    assert torch.count_nonzero(a.up.weight) == 0
    assert torch.count_nonzero(a.up.bias) == 0


def test_attaching_an_adapter_does_not_change_model_output():
    m = _Backbone()
    x = _x()
    before = m(x).clone()
    attach_ffn_adapter(m, bottleneck=4, attr="cohort_adapter")
    assert torch.equal(m(x), before), "attach must be a no-op until trained"


# 2. A trained adapter changes predictions

def test_a_perturbed_adapter_changes_the_output():
    m = _Backbone()
    x = _x()
    before = m(x).clone()
    a = attach_ffn_adapter(m, bottleneck=4, attr="cohort_adapter")
    with torch.no_grad():
        a.up.weight.normal_(0, 0.1)
    assert not torch.allclose(m(x), before), (
        "if this passes, the adapter is not in the forward path and every "
        "adapted method silently equals population_only"
    )


def test_adapter_parameters_receive_gradients():
    m = _Backbone()
    a = attach_ffn_adapter(m, bottleneck=4, attr="cohort_adapter")
    m(_x()).sum().backward()
    assert a.down.weight.grad is not None
    assert a.up.weight.grad is not None and a.up.weight.grad.abs().sum() > 0


def test_base_parameters_are_frozen_by_attach():
    m = _Backbone()
    attach_ffn_adapter(m, bottleneck=4, attr="cohort_adapter")
    assert not m.encoder.weight.requires_grad, "the backbone must stay frozen"


# 3. Cohort and patient compose

def test_both_adapters_apply_and_in_cohort_then_patient_order():
    m = _Backbone()
    x = _x()
    c = attach_ffn_adapter(m, bottleneck=4, attr="cohort_adapter")
    p = attach_ffn_adapter(m, bottleneck=4, attr="patient_adapter")
    with torch.no_grad():
        c.up.weight.fill_(0.01)
        cohort_only = m(x).clone()
        p.up.weight.fill_(0.02)
    both = m(x)
    assert not torch.allclose(both, cohort_only), "patient adapter must also apply"
    # Order matters: the patient adapter sees the cohort-adapted representation.
    m2 = copy.deepcopy(m)
    m2.cohort_adapter = None
    assert not torch.allclose(m2(x), both), "patient alone != cohort+patient"


def test_attach_is_idempotent_and_does_not_overwrite():
    m = _Backbone()
    a1 = attach_ffn_adapter(m, bottleneck=4, attr="cohort_adapter")
    with torch.no_grad():
        a1.up.weight.fill_(0.5)
    a2 = attach_ffn_adapter(m, bottleneck=4, attr="cohort_adapter")
    assert a2 is a1, "re-attaching must not silently discard trained weights"
    assert float(a2.up.weight[0, 0]) == pytest.approx(0.5)


# 4. Exactly two levels, and no way to add a third by accident

def test_the_backbone_has_exactly_two_adapter_slots():
    from cocofl.models.backbone import TFNFullModel
    import inspect
    src = inspect.getsource(TFNFullModel.forward)
    assert "self.cohort_adapter(" in src and "self.patient_adapter(" in src
    assert "ffn_adapter" not in src, (
        "a third slot applied after cohort and patient is not in the TD, which "
        "defines a cohort model per centre and a patient model per individual"
    )


def test_attr_is_required_so_no_call_can_land_in_the_wrong_slot():
    m = _Backbone()
    with pytest.raises(TypeError):
        attach_ffn_adapter(m, bottleneck=4)          # no attr -> refuse


def test_attr_is_keyword_only():
    import inspect
    p = inspect.signature(attach_ffn_adapter).parameters["attr"]
    assert p.kind is p.KEYWORD_ONLY and p.default is p.empty


def test_state_dict_helpers_also_require_an_explicit_slot():
    import inspect
    from cocofl.adapters.ffn import adapter_state_dict, load_adapter_state_dict
    for fn in (adapter_state_dict, load_adapter_state_dict):
        p = inspect.signature(fn).parameters["attr"]
        assert p.kind is p.KEYWORD_ONLY and p.default is p.empty, fn.__name__


# Position-wise: adapting every step then taking the last = adapting the last

def test_adapter_is_position_wise():
    """The head reads only hidden_states[:, -1, :], so adapting earlier steps changes nothing numerically; it only costs compute."""
    a = FFNAdapter(H, bottleneck=4)
    with torch.no_grad():
        a.up.weight.normal_(0, 0.1)
    x = _x()
    all_then_last = a(x)[:, -1, :]
    last_only = a(x[:, -1:, :])[:, -1, :]
    assert torch.allclose(all_then_last, last_only, atol=1e-6)
