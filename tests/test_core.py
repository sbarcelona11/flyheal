import copy
import numpy as np
import pytest
import torch
from gymnasium import spaces
from flyheal.models import FlyExtractor, connectivity
from flyheal.metrics import recovery_metrics


def test_random_control_preserves_density_and_degrees():
    structured = connectivity('structured', 64, 7)
    random = connectivity('random', 64, 7)
    assert structured.sum() == 1280
    assert np.array_equal(structured.sum(0), random.sum(0))
    assert np.array_equal(structured.sum(1), random.sum(1))
    assert not np.array_equal(structured, random)
    assert np.array_equal(random, connectivity('random', 64, 7))


@pytest.mark.parametrize('kind', ['dense', 'structured', 'random'])
@pytest.mark.parametrize('lesion', ['neurons', 'edges'])
def test_permanent_damage_survives_optimizer_and_serialization(kind, lesion):
    net = FlyExtractor(spaces.Box(-1, 1, (4,), dtype=np.float32), kind=kind, seed=3)
    before = net.effective_connections()
    details = net.lesion(lesion, 0.5, 11)
    assert details['removed'] > 0
    assert net.effective_connections() < before
    state = copy.deepcopy(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        net(torch.ones(8, 4)).square().mean().backward()
        optimizer.step()
    if lesion == 'neurons':
        assert torch.all(net(torch.ones(8, 4))[:, net.alive == 0] == 0)
    else:
        assert torch.all((net.layer2.weight * net.mask)[state['mask'] == 0] == 0)
    restored = FlyExtractor(net.observation_space, kind=kind, seed=99)
    restored.load_state_dict(net.state_dict())
    assert torch.equal(net(torch.ones(1, 4)), restored(torch.ones(1, 4)))


def test_zero_and_total_damage_and_invalid_fraction():
    net = FlyExtractor(spaces.Box(-1, 1, (4,), dtype=np.float32))
    x = torch.ones(2, 4)
    before = net(x).detach().clone()
    net.lesion('neurons', 0, 1)
    assert torch.equal(before, net(x))
    net.lesion('neurons', 1, 1)
    assert torch.all(net(x) == 0)
    with pytest.raises(ValueError):
        net.lesion('edges', 1.1, 1)
    with pytest.raises(ValueError):
        net.lesion('unknown', 0.5, 1)


def test_recovery_uses_lost_return_and_handles_no_drop():
    result = recovery_metrics(200, 100, 150, 250)
    assert result['drop'] == 100
    assert result['gain'] == 50
    assert result['recovered_fraction'] == 0.5
    assert result['healthy_gap'] == 100
    assert recovery_metrics(100, 120, 150, 160)['recovered_fraction'] is None
