import numpy as np
import pytest
import json
from pathlib import Path

from flyheal.environments import ForagingEnv


def test_contrast_amplifies_food_direction_without_changing_the_world():
    raw = ForagingEnv()
    contrast = ForagingEnv(sensor_encoding='contrast_v1')
    original, _ = raw.reset(seed=100000)
    encoded, _ = contrast.reset(seed=100000)
    raw_world, contrast_world = raw.snapshot(), contrast.snapshot()
    raw_world.pop('observation')
    contrast_world.pop('observation')
    assert contrast_world == raw_world
    np.testing.assert_array_equal(encoded[3:], original[3:])
    assert encoded[2] > encoded[1] > encoded[0]
    assert encoded[2] - encoded[0] > 5 * (original[2] - original[0])
    assert contrast.observation_space.contains(encoded)
    # The preprocessing must not alter transitions or rewards.
    for action in [0, 1, 2] * 10:
        a = raw.step(action)
        b = contrast.step(action)
        assert a[1:] == b[1:]
        np.testing.assert_array_equal(a[0][3:], b[0][3:])


def test_contrast_does_not_reveal_food_beyond_sensor_range():
    env = ForagingEnv(sensor_encoding='contrast_v1')
    env.reset(seed=0)
    env.position = np.array([5., 5.])
    env.foods = np.array([[.5, .5], [9.5, 9.5], [.5, 9.5]])
    np.testing.assert_array_equal(env._observation()[:3], [0., 0., 0.])


def test_invalid_sensor_encoding_is_rejected():
    with pytest.raises(ValueError, match='sensor'):
        ForagingEnv(sensor_encoding='unknown')


def test_experiment_and_replay_use_the_recorded_sensor_encoding(tmp_path):
    from flyheal.experiment import run
    from flyheal.ui import ReplayStore

    config = json.loads(Path('configs/smoke.json').read_text())
    config.update(sensor_encoding='contrast_v1', seeds=[0], eval_episodes=1)
    run(config, tmp_path)
    replay = ReplayStore(tmp_path).replay('structured', 0, 'neurons_0.5', 'baseline')
    expected, _ = ForagingEnv(sensor_encoding='contrast_v1').reset(seed=100000)
    np.testing.assert_allclose(replay['frames'][0]['state'], expected)


def test_experiment_rejects_unknown_encoding_before_training():
    from flyheal.experiment import validate_config

    config = json.loads(Path('configs/smoke.json').read_text())
    config['sensor_encoding'] = 'unknown'
    with pytest.raises(ValueError, match='sensor'):
        validate_config(config)
