"""Local presentation UI backed by real checkpoint inference and recorded metrics."""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import torch
from stable_baselines3 import PPO
from .environments import make_env


class ReplayStore:
    def __init__(self, results):
        self.results = Path(results).resolve()
        self.meta = json.loads((self.results / 'metadata.json').read_text())
        if self.meta['status'] != 'complete':
            raise ValueError('La interfaz necesita una corrida completa')
        self.config = self.meta['config']
        if self.config.get('environment', 'foraging') != 'foraging':
            raise ValueError('Solo está disponible el mundo 2D')
        self.conditions = [f'{mode}_{fraction:g}' for mode in self.config['lesions']
                           for fraction in self.config['fractions']]
        self.cache = {}
        self.lock = threading.Lock()
        torch.set_num_threads(1)

    def summary(self):
        raw = pd.read_csv(self.results / 'episodes.csv')
        series = raw.groupby(['architecture', 'seed', 'condition', 'steps']).reward.mean().reset_index()
        task_path = self.results / 'task_metrics.csv'
        task = json.loads(pd.read_csv(task_path).to_json(orient='records')) if task_path.exists() else []
        return {'config': self.config, 'task_metrics': task, 'conditions': self.conditions,
                'run': self.results.name, 'provenance': ('malecns-measured-subgraph' if 'male_cns' in self.config['architectures'] else 'synthetic'),
                'elapsed_seconds': self.meta['elapsed_seconds'],
                'seeds': json.loads(pd.read_csv(self.results / 'summary.csv').to_json(orient='records')),
                'aggregate': json.loads(pd.read_csv(self.results / 'aggregate.csv').to_json(orient='records')),
                'series': json.loads(series.to_json(orient='records'))}

    def replay(self, architecture, seed, condition, stage):
        if (architecture not in self.config['architectures'] or seed not in self.config['seeds']
                or condition not in self.conditions or stage not in ('baseline', 'post', 'recovered')):
            raise ValueError('Selección inválida de arquitectura, semilla, lesión o etapa')
        key = (architecture, seed, condition if stage != 'baseline' else '', stage)
        with self.lock:
            if key not in self.cache:
                self.cache[key] = self._calculate(architecture, seed, condition, stage)
            return {**self.cache[key], 'condition': condition}

    def _calculate(self, architecture, seed, condition, stage):
        filename = 'baseline.zip' if stage == 'baseline' else f'{condition}_{stage}.zip'
        model = PPO.load(self.results / 'checkpoints' / f'{architecture}_seed{seed}' / filename, device='cpu')
        model.policy.set_training_mode(False)
        extractor = model.policy.features_extractor
        if stage == 'baseline':
            original_mask = extractor.mask.cpu().int().tolist()
        else:
            original_model = PPO.load(self.results / 'checkpoints' / f'{architecture}_seed{seed}' / 'baseline.zip', device='cpu')
            original_mask = original_model.policy.features_extractor.mask.cpu().int().tolist()
            del original_model
        environment = self.config.get('environment', 'foraging')
        env = make_env(environment)
        frames = []
        evaluation_seed = 100000 + seed * 100
        try:
            observation, _ = env.reset(seed=evaluation_seed)
            terminated = truncated = False
            total, steps = 0.0, 0
            info = {}
            while True:
                tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    first, final = extractor.activations(tensor)
                    h1, h2 = first[0], final[0]
                    logits = model.policy.action_net(final)[0]
                    probabilities = torch.softmax(logits, dim=0)
                frames.append({'step': steps, 'state': observation.tolist(),
                               'h1': [round(float(x), 6) for x in h1],
                               'h2': [round(float(x), 6) for x in h2],
                               'decision': {'logits': logits.tolist(), 'probabilities': probabilities.tolist()}})
                if environment == 'foraging':
                    frames[-1]['world'] = env.snapshot()
                if terminated or truncated:
                    break
                action, _ = model.predict(observation, deterministic=True)
                frames[-1]['action'] = int(action)
                observation, reward, terminated, truncated, info = env.step(int(action))
                total += float(reward)
                steps += 1
        finally:
            env.close()
        real = hasattr(extractor, 'cell_metadata')
        provenance = ('malecns-rewired-control' if extractor.rewired else 'malecns-measured-subgraph') if real else 'synthetic'
        return {'decision': {'architecture': 'linear',
                             'weights': model.policy.action_net.weight.detach().cpu().tolist(),
                             'bias': model.policy.action_net.bias.detach().cpu().tolist(),
                             'actions': ['Girar izquierda', 'Avanzar', 'Girar derecha'] if environment == 'foraging' else ['Izquierda', 'Derecha']},
                'population': 'single' if real else 'two-layers',
                'cells': extractor.cell_metadata if real else None,
                'graph_sha256': extractor.graph_sha256 if real else None,
                'modeled_connections': extractor.modeled_connections() if real else extractor.effective_connections(),
                'architecture': architecture, 'seed': seed, 'condition': condition,
                'stage': stage, 'eval_seed': evaluation_seed,
                'environment': environment, 'task_metrics': info if environment == 'foraging' else {}, 'provenance': provenance,
                'return': total, 'terminated': bool(terminated), 'truncated': bool(truncated),
                'frames': frames, 'alive': extractor.alive.cpu().int().tolist(),
                'mask': extractor.mask.cpu().int().tolist(), 'original_mask': original_mask,
                'active_connections': extractor.effective_connections()}


def make_server(results, port=8765, training_root=None, host='127.0.0.1'):
    from .training import TrainingManager
    manager = TrainingManager(results, training_root)
    web_root = Path(__file__).parent / 'web'
    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, status=200, content_type='application/json; charset=utf-8'):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            try:
                params = parse_qs(url.query)
                if any(len(values) != 1 for values in params.values()):
                    raise ValueError('Parámetros duplicados')
                if url.path == '/api/atlas':
                    from .connectome import GRAPH_PATH
                    return self.respond((GRAPH_PATH.parent / 'atlas.json').read_bytes())
                if url.path == '/api/runs':
                    return self.respond(manager.runs())
                if url.path == '/api/progress':
                    return self.respond(manager.status())
                if url.path == '/api/summary':
                    return self.respond(manager.store(params.get('run', [None])[0]).summary())
                if url.path == '/api/replay':
                    params = parse_qs(url.query)
                    if any(len(values) != 1 for values in params.values()):
                        raise ValueError('Parámetros duplicados')
                    selection = {key: params[key][0] for key in ['architecture', 'seed', 'condition', 'stage']}
                    selection['seed'] = int(selection['seed'])
                    return self.respond(manager.store(params.get('run', [None])[0]).replay(**selection))
                files = {'/': 'index.html', '/index.html': 'index.html',
                         '/app.js': 'app.js', '/playback.mjs': 'playback.mjs', '/brain.mjs': 'brain.mjs', '/decision.mjs': 'decision.mjs', '/styles.css': 'styles.css'}
                if url.path not in files:
                    return self.respond({'error': 'Recurso no disponible'}, 404)
                name = files[url.path]
                mime = {'html': 'text/html', 'js': 'text/javascript', 'mjs': 'text/javascript', 'css': 'text/css'}[name.split('.')[-1]]
                return self.respond((web_root / name).read_bytes(), content_type=mime + '; charset=utf-8')
            except (ValueError, KeyError) as error:
                self.respond({'error': str(error)}, 400)
            except Exception:
                self.respond({'error': 'No se pudo cargar el modelo. Revisá la carpeta de resultados.'}, 500)

        def do_POST(self):
            origins = {f'http://127.0.0.1:{self.server.server_port}',
                       f'http://localhost:{self.server.server_port}'}
            if ('http://' + self.headers.get('Host', '') not in origins
                    or self.headers.get('Origin', next(iter(origins))) not in origins):
                return self.respond({'error': 'Solo se permiten solicitudes locales'}, 403)
            try:
                if self.path == '/api/cancel':
                    return self.respond(manager.cancel())
                if self.path != '/api/train':
                    return self.respond({'error': 'Recurso no disponible'}, 404)
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Se necesita una configuración JSON')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('Tamaño de configuración inválido')
                options = json.loads(self.rfile.read(length))
                if not isinstance(options, dict):
                    raise ValueError('La configuración debe ser un objeto')
                return self.respond(manager.start(options), 202)
            except (ValueError, TypeError) as error:
                self.respond({'error': str(error)}, 400)
            except RuntimeError as error:
                self.respond({'error': str(error)}, 409)

        def log_message(self, format, *args):
            pass
    class Server(ThreadingHTTPServer):
        def server_close(self):
            manager.close()
            super().server_close()
    return Server((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description='FlyHeal local visual laboratory')
    default_results = Path('results/new_run')
    parser.add_argument('--results', type=Path, default=default_results)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1', help='Interfaz de escucha (0.0.0.0 para Docker)')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    server = make_server(args.results, args.port, host=args.host)
    address = f'http://127.0.0.1:{server.server_port}'
    print(f'FlyHeal disponible en {address} · Ctrl+C para cerrar', flush=True)
    if not args.no_open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
