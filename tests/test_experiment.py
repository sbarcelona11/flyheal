import json
from pathlib import Path
import numpy as np
import pytest
from stable_baselines3 import PPO
from flyheal.experiment import run, validate_config


def test_runner_emits_paired_returns_and_reloadable_damaged_policy(tmp_path):
    config = json.loads(Path('configs/smoke.json').read_text())
    run(config, tmp_path)
    import pandas as pd
    data = pd.read_csv(tmp_path / 'episodes.csv')
    assert {'baseline', 'healthy', 'neurons_0.5', 'edges_0.5'} == set(data.condition)
    for _, group in data.groupby(['architecture', 'seed', 'condition', 'steps']):
        assert len(group) == 2
        assert group.eval_seed.tolist() == [100000, 100001]
    pre = PPO.load(tmp_path / 'checkpoints/structured_seed0/baseline.zip', device='cpu')
    post = PPO.load(tmp_path / 'checkpoints/structured_seed0/neurons_0.5_post.zip', device='cpu')
    recovered = PPO.load(tmp_path / 'checkpoints/structured_seed0/neurons_0.5_recovered.zip', device='cpu')
    assert pre.policy.features_extractor.alive.sum().item() == 64
    assert post.policy.features_extractor.alive.sum().item() == 32
    assert recovered.policy.features_extractor.alive.sum().item() == 32
    obs = np.zeros(8, dtype=np.float32)
    assert recovered.predict(obs, deterministic=True)[0] in (0, 1, 2)
    assert (tmp_path / 'summary.csv').is_file()
    assert (tmp_path / 'metadata.json').is_file()


def test_budget_validation_rejects_ppo_rounding():
    config = json.loads(Path('configs/smoke.json').read_text())
    config['checkpoint_steps'] = 129
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize('field,value', [
    ('architectures', ['dense', 'dense']),
    ('lesions', ['neurons', 'neurons']),
    ('fractions', [0.5, 0.5]),
    ('fractions', [0.50000001, 0.50000002]),
])
def test_duplicate_conditions_are_rejected(field, value):
    config = json.loads(Path('configs/smoke.json').read_text())
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)
