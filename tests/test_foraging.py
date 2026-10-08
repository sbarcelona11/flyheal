import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from flyheal.environments import ForagingEnv, make_env
from flyheal.experiment import run


def test_seeded_maps_and_trajectories():
    a, b = ForagingEnv(), ForagingEnv()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert a.snapshot() == b.snapshot()
    for action in [0, 1, 2] * 20:
        ra, rb = a.step(action), b.step(action)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1:] == rb[1:]
        assert a.observation_space.contains(ra[0])
    b.reset(seed=8)
    assert a.snapshot()['foods'] != b.snapshot()['foods']
    assert oa.shape == ob.shape == (8,)


def test_food_collision_and_horizon():
    env = ForagingEnv()
    env.reset(seed=0)
    env.obstacles = []
    env.position = np.array([5., 5.])
    env.heading = 0.
    env.foods = np.array([[5.18, 5.], [8., 8.], [2., 2.]])
    env.collected = np.zeros(3, dtype=bool)
    _, reward, terminated, truncated, info = env.step(1)
    assert info['foods_collected'] == 1 and reward > 9 and not terminated
    env.position = np.array([9.8, 5.])
    before = env.position.copy()
    _, _, _, _, info = env.step(1)
    np.testing.assert_equal(before, env.position)
    assert info['collisions'] == 1
    env.position = np.array([4.8, 5.])
    env.obstacles = [(5., 4., 1., 2.)]
    env.heading = 0.
    before = env.position.copy()
    env.step(1)
    np.testing.assert_equal(before, env.position)
    assert env.collisions == 2
    env.obstacles = []
    env.steps = 499
    assert env.step(0)[3]
    env.position = np.array([8., 8.])
    env.collected[:] = [True, False, True]
    _, _, terminated, _, info = env.step(1)
    assert terminated and info['success']
    json.dumps(env.snapshot())


def test_sensors_are_local_and_do_not_expose_food_coordinates():
    env = ForagingEnv()
    env.reset(seed=0)
    env.position = np.array([5., 5.])
    env.foods = np.array([[.5, .5], [9.5, 9.5], [.5, 9.5]])
    obs = env._observation()
    assert np.all(obs[:3] == 0)
    assert obs.shape == (8,) and env.observation_space.contains(obs)
    assert make_env().observation_space.shape == (8,)


def test_real_ppo_foraging_branches_and_metrics(tmp_path):
    config = json.loads(Path('configs/smoke.json').read_text())
    config.update(environment='foraging', eval_episodes=1)
    run(config, tmp_path)
    import pandas as pd
    raw = pd.read_csv(tmp_path / 'episodes.csv')
    assert {'foods_collected', 'collisions', 'success'} <= set(raw.columns)
    assert (tmp_path / 'task_metrics.csv').exists()
    from flyheal.plots import plot_results
    from flyheal.report import build_report
    plot_results(tmp_path)
    report = build_report(tmp_path)
    assert 'Comida y colisiones' in report
    assert 'resolver CartPole' not in report
    assert 'Retornos medios sin descontar, máximo 500.' not in report
    assert (tmp_path / 'figures/baseline.png').exists()
    for mode in ['neurons', 'edges']:
        model = PPO.load(tmp_path / f'checkpoints/structured_seed0/{mode}_0.5_recovered.zip')
        env = make_env('foraging')
        obs, _ = env.reset(seed=3)
        assert int(model.predict(obs)[0]) in (0, 1, 2)
        ext = model.policy.features_extractor
        if mode == 'neurons':
            assert int(ext.alive.sum()) == 32
        else:
            post = PPO.load(tmp_path / f'checkpoints/structured_seed0/{mode}_0.5_post.zip')
            np.testing.assert_equal(ext.mask.numpy(), post.policy.features_extractor.mask.numpy())


def test_gym_contract():
    from gymnasium.utils.env_checker import check_env
    check_env(ForagingEnv(), skip_render_check=True)


def test_training_config_environment_default_and_validation():
    from flyheal.training import training_config
    import pytest
    assert training_config({})['environment'] == 'foraging'
    assert training_config({'environment': 'foraging'})['environment'] == 'foraging'
    with pytest.raises(ValueError):
        training_config({'environment': 'invalid'})


def test_signed_returns_do_not_report_90_percent_recovery():
    import pandas as pd
    from flyheal.metrics import summarize
    rows = []
    for condition, points in [('baseline', [(0, -10), (512, -5)]),
                              ('healthy', [(0, -5), (512, -4)]),
                              ('neurons_0.5', [(0, -8), (512, -4)])]:
        for steps, reward in points:
            rows.append(dict(architecture='structured', seed=0, condition=condition, steps=steps, reward=reward))
    assert pd.isna(summarize(pd.DataFrame(rows)).iloc[0].steps_to_90pct)
