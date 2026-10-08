"""Paired PPO experiments with raw episode logging and independent branches."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from .connectome import ConnectomeExtractor, REAL_KINDS, load_graph, GRAPH_PATH
from .environments import make_env
from .models import FlyExtractor, KINDS
from .metrics import summarize, aggregate


def validate_config(config):
    if config.get('environment', 'foraging') != 'foraging':
        raise ValueError('Unknown environment')
    required = ['seeds', 'architectures', 'width', 'baseline_steps', 'recovery_steps',
                'checkpoint_steps', 'eval_episodes', 'lesions', 'fractions', 'ppo']
    if any(key not in config for key in required):
        raise ValueError('Incomplete experiment configuration')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Nonempty unique seeds required')
    if not config['architectures'] or not set(config['architectures']) <= set(KINDS):
        raise ValueError('Unknown architecture')
    if set(config['architectures']) & set(REAL_KINDS):
        if config.get('environment', 'foraging') != 'foraging' or config['width'] != 80:
            raise ValueError('MaleCNS requires foraging and fixed width80')
        load_graph()
    if not config['lesions'] or not set(config['lesions']) <= {'neurons', 'edges'}:
        raise ValueError('Unknown lesion')
    if not config['fractions'] or any(not 0 <= f <= 1 for f in config['fractions']):
        raise ValueError('Fractions must be in [0,1]')
    for field in ('architectures', 'lesions'):
        if len(set(config[field])) != len(config[field]):
            raise ValueError(f'{field} must contain unique values')
    names = [f'{mode}_{fraction:g}' for mode in config['lesions'] for fraction in config['fractions']]
    if len(set(names)) != len(names):
        raise ValueError('Lesion condition names must be unique, including formatted fractions')
    if config.get('density') is not None and not 0 < config['density'] <= 1:
        raise ValueError('density must be in (0,1]')
    if config['width'] < 16 or config['width'] % 4 or config['eval_episodes'] < 1:
        raise ValueError('Invalid width or evaluation budget')
    rollout = config['ppo'].get('n_steps', 512)
    checkpoint = config['checkpoint_steps']
    if checkpoint <= 0 or checkpoint % rollout:
        raise ValueError('checkpoint_steps must be a positive multiple of PPO n_steps')
    for field in ('baseline_steps', 'recovery_steps'):
        if config[field] <= 0 or config[field] % checkpoint:
            raise ValueError(f'{field} must be a positive multiple of checkpoint_steps')


def evaluate(model, episodes, seed, environment='foraging'):
    """Greedy evaluation on a separate env, no training RNG or env mutation."""
    env = make_env(environment)
    results = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total, terminated, truncated = 0.0, False, False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total += reward
            row = (seed + episode, total, bool(truncated))
            results.append(row + (info,) if environment == 'foraging' else row)
    finally:
        env.close()
    return results


def reset_branch_optimizer(model):
    """Reset Adam moments symmetrically for damaged and healthy continuation."""
    policy = model.policy
    policy.optimizer = policy.optimizer_class(policy.parameters(),
                                             lr=model.learning_rate,
                                             **policy.optimizer_kwargs)


def run(config, output):
    validate_config(config)
    environment = config.get('environment', 'foraging')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metadata.json').exists():
        raise FileExistsError('Use a new results directory; existing run is preserved')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    packages = ['torch', 'numpy', 'gymnasium', 'stable-baselines3', 'pandas', 'matplotlib']
    meta = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
            'config': config, 'python': platform.python_version(),
            'platform': platform.platform(), 'device': 'cpu', 'torch_threads': 1,
            'packages': {p: importlib.metadata.version(p) for p in packages},
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in Path(__file__).parent.glob('*.py')},
            'graph_provenance': 'synthetic, connectome-inspired; no MaleCNS data',
            'evaluation': 'deterministic, paired initial states, 100000 + seed*100 + episode',
            'optimizer_continuation': 'reset identically in all branches',
            'lesions': []}
    def write_meta():
        temp = output / 'metadata.tmp'
        temp.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        temp.replace(output / 'metadata.json')
    if set(config['architectures']) & set(REAL_KINDS):
        graph = load_graph()
        meta['graph_provenance'] = 'Validated real MaleCNS induced 80-cell subset; dense capacity baseline and degree-preserving rewired control are synthetic. Dynamics, transmitter signs, sensor injection and policy readout are modeling assumptions.'
        meta['connectome'] = {'graph_sha256': hashlib.sha256(GRAPH_PATH.read_bytes()).hexdigest(),
            'cells': len(graph['nodes']), 'measured_directed_edges': len(graph['edges']),
            'contacts': sum(edge[2] for edge in graph['edges']),
            'validation': json.loads(GRAPH_PATH.with_name('validation.json').read_text()),
            'iterations': 5, 'leak': .5, 'state_reset_each_decision': True,
            'sensor_mapping': 'eight foraging sensors assigned to the eight selected visual cell-type channels; engineering assumption',
            'readout': 'learned PPO actor and critic from all 80 modeled cell activities',
            'normalization': 'contact count divided by total incoming anatomical contacts; assumed presynaptic transmitter sign',
            'sign_zero': 'anatomical edges retained, modeled synaptic drive zero'}
    write_meta()
    snapshot = output / 'source_used'
    snapshot.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob('*.py'):
        (snapshot / source.name).write_bytes(source.read_bytes())
    rows = []
    total_training_steps = 0
    target_steps = len(config['seeds']) * len(config['architectures']) * (
        config['baseline_steps'] + config['recovery_steps'] * (1 + len(config['lesions']) * len(config['fractions'])))
    def progress(arch, seed, condition, steps, mean=None):
        info = {'architecture': arch, 'seed': seed, 'condition': condition,
                'local_steps': steps, 'trained_steps': total_training_steps,
                'target_steps': target_steps, 'percent': min(99, round(100 * total_training_steps / target_steps, 1)),
                'last_return': mean, 'elapsed_seconds': time.perf_counter() - start}
        temp = output / 'progress.tmp'
        temp.write_text(json.dumps(info), encoding='utf-8')
        temp.replace(output / 'progress.json')
    progress('', 0, 'preparing', 0)
    def record(model, arch, seed, condition, steps):
        for result in evaluate(model, config['eval_episodes'], 100000 + seed * 100, environment):
            eval_seed, reward, truncated = result[:3]
            row = {'architecture': arch, 'seed': seed, 'condition': condition,
                   'steps': steps, 'eval_seed': eval_seed, 'reward': reward, 'time_limit': truncated}
            if environment == 'foraging':
                row.update({k: result[3][k] for k in ('foods_collected', 'collisions', 'success')})
            rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'episodes.csv', index=False)
        recent = [r['reward'] for r in rows[-config['eval_episodes']:]]
        progress(arch, seed, condition, steps, float(np.mean(recent)))
        print(f'{arch:10s} seed={seed} {condition:12s} steps={steps:6d} return={np.mean(recent):6.1f}', flush=True)

    for arch in config['architectures']:
        for seed in config['seeds']:
            folder = output / 'checkpoints' / f'{arch}_seed{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            env = make_env(environment)
            model = PPO('MlpPolicy', env, seed=seed, device='cpu', verbose=0,
                        policy_kwargs={'features_extractor_class': ConnectomeExtractor if arch in REAL_KINDS else FlyExtractor,
                                       'features_extractor_kwargs': {'kind': arch, 'width': config['width'], 'seed': seed, 'density': config.get('density')},
                                       'net_arch': [], 'share_features_extractor': True},
                        **config['ppo'])
            record(model, arch, seed, 'baseline', 0)
            for steps in range(config['checkpoint_steps'], config['baseline_steps'] + 1, config['checkpoint_steps']):
                progress(arch, seed, 'baseline', steps - config['checkpoint_steps'])
                model.learn(config['checkpoint_steps'], reset_num_timesteps=False)
                total_training_steps += config['checkpoint_steps']
                record(model, arch, seed, 'baseline', steps)
            model.save(folder / 'baseline.zip')
            extractor = model.policy.features_extractor
            meta.setdefault('architectures', []).append({
                'architecture': arch, 'seed': seed,
                'stored_parameters': sum(p.numel() for p in model.policy.parameters()),
                'active_layer2_connections': extractor.effective_connections(),
                'functional_parameters_pre': extractor.modeled_connections() + 8 + 80 + 4*80 + 4 if arch in REAL_KINDS else (int(extractor.mask.sum()) + 14 * config['width'] + 4)})
            if arch in REAL_KINDS:
                meta['architectures'][-1].update(
                    anatomical_connections=extractor.effective_connections(), modeled_connections=extractor.modeled_connections(),
                    population='single 80-cell measured population',
                    topology='measured MaleCNS' if arch == 'male_cns' else 'degree-preserving rewiring of measured MaleCNS',
                    gain_parameterization='positive exp(clamp(log_gain,-3,3)) per anatomical edge')
            env.close()
            del model
            conditions = [('healthy', None, 0)] + [
                (f'{mode}_{fraction:g}', mode, fraction)
                for mode in config['lesions'] for fraction in config['fractions']]
            for condition, mode, fraction in conditions:
                branch_env = make_env(environment)
                branch = PPO.load(folder / 'baseline.zip', env=branch_env, device='cpu')
                branch.set_random_seed(20000 + seed)
                reset_branch_optimizer(branch)
                extractor = branch.policy.features_extractor
                if mode:
                    details = extractor.lesion(mode, fraction, 30000 + seed + (1000 if mode == 'edges' else 0))
                    meta['lesions'].append({'architecture': arch, 'training_seed': seed,
                                            'condition': condition, **details})
                branch.save(folder / f'{condition}_post.zip')
                record(branch, arch, seed, condition, 0)
                for steps in range(config['checkpoint_steps'], config['recovery_steps'] + 1, config['checkpoint_steps']):
                    progress(arch, seed, condition, steps - config['checkpoint_steps'])
                    branch.learn(config['checkpoint_steps'], reset_num_timesteps=False)
                    total_training_steps += config['checkpoint_steps']
                    record(branch, arch, seed, condition, steps)
                branch.save(folder / f'{condition}_recovered.zip')
                np.savez_compressed(folder / f'{condition}_masks.npz',
                                    mask=extractor.mask.cpu().numpy(), alive=extractor.alive.cpu().numpy())
                branch_env.close()
                del branch
            write_meta()
    frame = pd.DataFrame(rows)
    if environment == 'foraging':
        keys = ['architecture', 'seed', 'condition', 'steps']
        task = frame.groupby(keys)[['foods_collected', 'collisions', 'success']].mean().reset_index()
        task = task.rename(columns={'foods_collected': 'mean_foods', 'collisions': 'mean_collisions', 'success': 'success_rate'})
        def stage(row):
            if row.condition == 'baseline':
                return 'pre' if row.steps == config['baseline_steps'] else 'baseline'
            if row.condition == 'healthy':
                return 'healthy'
            return 'post' if row.steps == 0 else ('recovered' if row.steps == config['recovery_steps'] else 'retraining')
        task['stage'] = task.apply(stage, axis=1)
        task.to_csv(output / 'task_metrics.csv', index=False)
    summary = summarize(frame)
    summary.to_csv(output / 'summary.csv', index=False)
    aggregate(summary).to_csv(output / 'aggregate.csv', index=False)
    meta.update(status='complete', elapsed_seconds=time.perf_counter() - start,
                total_training_steps=total_training_steps, evaluation_episodes=len(rows),
                finished_utc=datetime.now(timezone.utc).isoformat())
    write_meta()
    return summary


def main():
    parser = argparse.ArgumentParser(description='FlyHeal CPU experiment')
    parser.add_argument('--config', type=Path, default=Path('configs/malecns_quick.json'))
    parser.add_argument('--out', type=Path, default=Path('results/new_run'))
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out)
    from .plots import plot_results
    plot_results(args.out)


if __name__ == '__main__':
    main()
