"""Bounded CPU training jobs launched from the local web interface."""
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

BUDGETS = {'fast': (20480, 10240), 'standard': (40960, 20480), 'extended': (102400, 40960)}
WIDTHS = (64, 80, 128, 256)
DENSITIES = (.125, .25, .3125, .5, .75, 1.)


def training_config(options):
    environment = options.get('environment', 'foraging')
    if environment != 'foraging':
        raise ValueError('Entorno inválido')
    architecture = options.get('architecture', 'structured')
    real = architecture in ('male_cns', 'rewired_male_cns')
    width = options.get('width', 80 if real else 64)
    density = options.get('density', .3125)
    budget = options.get('budget', 'fast')
    seeds = options.get('seeds', 1)
    if architecture not in ('dense', 'structured', 'random', 'male_cns', 'rewired_male_cns') or type(width) is not int or width not in WIDTHS:
        raise ValueError('Arquitectura inválida o ancho fuera de 64/80/128/256')
    if real and (width != 80 or environment != 'foraging'):
        raise ValueError('MaleCNS requiere mundo 2D y 80 células')
    if (not real and density not in DENSITIES) or budget not in BUDGETS or type(seeds) is not int or seeds not in (1, 3):
        raise ValueError('Conectividad, presupuesto o número de semillas inválido')
    baseline, recovery = BUDGETS[budget]
    config = {'environment': environment, 'architectures': [architecture], 'width': width,
            'density': None if real else (1. if architecture == 'dense' else density),
            'seeds': list(range(seeds)), 'baseline_steps': baseline, 'recovery_steps': recovery,
            'checkpoint_steps': 5120, 'eval_episodes': 20,
            'lesions': ['neurons', 'edges'], 'fractions': [.5],
            'ppo': {'n_steps': 512, 'batch_size': 64, 'n_epochs': 5,
                    'learning_rate': .0003, 'gamma': .99, 'gae_lambda': .95, 'ent_coef': 0.}}
    if real:
        from .connectome import load_graph, GRAPH_PATH
        import hashlib
        graph = load_graph()
        config['connectome'] = {'asset': 'malecns80', 'graph_sha256': hashlib.sha256(GRAPH_PATH.read_bytes()).hexdigest(),
                               'cells': len(graph['nodes']), 'edges': len(graph['edges']), 'validation': 'passed'}
    return config


def write_json(path, value):
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


class TrainingManager:
    def __init__(self, base_results, training_root=None):
        self.base = Path(base_results).resolve()
        self.root = Path(training_root).resolve() if training_root else self.base.parent / 'web'
        self.process = None
        self.job = None
        self.log_file = None
        self.lock = threading.RLock()
        self.stores = {}

    def _paths(self):
        paths = {self.base.name: self.base}
        if self.base.parent.exists():
            for candidate in self.base.parent.iterdir():
                if (candidate / 'metadata.json').is_file():
                    paths[candidate.name] = candidate
        if self.root.exists():
            for candidate in self.root.iterdir():
                path = candidate / 'metadata.json'
                job = candidate / 'job.json'
                if path.is_file() and job.is_file():
                    try:
                        if json.loads(job.read_text()).get('status') == 'complete':
                            paths[candidate.name] = candidate
                    except (OSError, ValueError):
                        continue
        return paths

    def runs(self):
        self.status()
        rows = []
        for name, path in self._paths().items():
            try:
                meta = json.loads((path / 'metadata.json').read_text())
                if meta['status'] != 'complete' or meta['config'].get('environment', 'foraging') != 'foraging':
                    continue
                cfg = meta['config']
                rows.append({'id': name, 'environment': cfg.get('environment', 'foraging'), 'width': cfg['width'], 'density': cfg.get('density', .3125),
                             'architectures': cfg['architectures'], 'seeds': cfg['seeds'],
                             'baseline_steps': cfg['baseline_steps']})
            except (OSError, ValueError, KeyError):
                continue
        default = self.base.name if any(row['id'] == self.base.name for row in rows) else (rows[0]['id'] if rows else None)
        return {'default': default, 'runs': rows}

    def store(self, name=None):
        from .ui import ReplayStore
        name = name or self.runs()['default']
        with self.lock:
            self.status()
            paths = self._paths()
            if name not in paths:
                raise ValueError('Corrida no disponible o todavía incompleta')
            if name not in self.stores:
                self.stores[name] = ReplayStore(paths[name])
            return self.stores[name]

    def start(self, options):
        return self.start_config(training_config(options))

    def start_config(self, config):
        from .experiment import validate_config
        validate_config(config)
        with self.lock:
            self.status()
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError('Ya hay un entrenamiento en curso')
            identifier = 'web_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
            directory = self.root / identifier
            directory.mkdir(parents=True)
            write_json(directory / 'config.json', config)
            self.job = {'id': identifier, 'status': 'running', 'config': config}
            write_json(directory / 'job.json', self.job)
            self.log_file = (directory / 'run.log').open('w', encoding='utf-8')
            env = os.environ.copy()
            env['FLYHEAL_PARENT_PID'] = str(os.getpid())
            package_root = str(Path(__file__).resolve().parent.parent)
            env['PYTHONPATH'] = package_root + os.pathsep + env.get('PYTHONPATH', '')
            try:
                self.process = subprocess.Popen([sys.executable, '-u', '-m', 'flyheal.train_worker',
                                                 str(directory / 'config.json'), str(directory)],
                                                cwd=package_root, env=env,
                                                stdout=self.log_file, stderr=subprocess.STDOUT)
            except Exception:
                self.log_file.close()
                self.job['status'] = 'failed'
                write_json(directory / 'job.json', self.job)
                raise
            self.job['worker_pid'] = self.process.pid
            write_json(directory / 'job.json', self.job)
            return dict(self.job)

    def status(self):
        with self.lock:
            if self.job is None:
                return {'status': 'idle'}
            directory = self.root / self.job['id']
            if self.process is not None and self.process.poll() is not None and self.job['status'] == 'running':
                self.job['status'] = 'complete' if self.process.returncode == 0 else 'failed'
                self.log_file.close()
                write_json(directory / 'job.json', self.job)
            result = dict(self.job)
            try:
                result['progress'] = json.loads((directory / 'progress.json').read_text())
            except (OSError, ValueError):
                cfg = self.job['config']
                target = len(cfg['seeds'])*len(cfg['architectures'])*(cfg['baseline_steps'] + cfg['recovery_steps']*(1+len(cfg['lesions'])*len(cfg['fractions'])))
                result['progress'] = {'percent': 0, 'trained_steps': 0, 'target_steps': target, 'condition': 'preparing'}
            if result['status'] == 'failed':
                try: result['error'] = (directory / 'run.log').read_text()[-1200:]
                except OSError: result['error'] = 'No se pudo iniciar el entrenamiento'
            return result

    def cancel(self):
        with self.lock:
            self.status()
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
                self.log_file.close()
                self.job['status'] = 'cancelled'
                write_json(self.root / self.job['id'] / 'job.json', self.job)
            return self.status()

    def close(self):
        self.cancel()
