import pytest
from torch import nn


@pytest.fixture
def trunc_normal_stds(monkeypatch) -> dict:
    """Patches nn.init.trunc_normal_ to record the std it gives each tensor: {id(tensor): std}, filled as it runs."""
    drawn_stds = {}
    trunc_normal = nn.init.trunc_normal_

    def recording_trunc_normal(tensor, *args, **kwargs):
        drawn_stds[id(tensor)] = kwargs.get("std")
        return trunc_normal(tensor, *args, **kwargs)

    monkeypatch.setattr(nn.init, "trunc_normal_", recording_trunc_normal)
    return drawn_stds
