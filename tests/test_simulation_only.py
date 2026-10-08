import pytest
from flyheal.training import TrainingManager, training_config
from flyheal.environments import make_env


def test_training_defaults_to_simulation_and_rejects_removed_environment():
    assert training_config({})['environment'] == 'foraging'
    with pytest.raises(ValueError):
        training_config({'environment': 'cartpole'})
    with pytest.raises(ValueError):
        make_env('cartpole')


def test_empty_project_has_no_default_or_saved_runs(tmp_path):
    manager = TrainingManager(tmp_path / 'new_run', tmp_path / 'web')
    assert manager.runs() == {'default': None, 'runs': []}
    assert manager.status() == {'status': 'idle'}
    with pytest.raises(ValueError):
        manager.store()


def test_empty_server_serves_training_and_empty_run_list(tmp_path):
    import json
    import threading
    import urllib.request
    from flyheal.ui import make_server
    server = make_server(tmp_path / 'new_run', port=0, training_root=tmp_path / 'web')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(address + '/api/runs') as response:
            assert json.load(response) == {'default': None, 'runs': []}
        with urllib.request.urlopen(address) as response:
            html = response.read().decode()
            assert 'id="emptyState"' in html
            assert 'value="cartpole"' not in html
        with urllib.request.urlopen(address + '/api/progress') as response:
            assert json.load(response)['status'] == 'idle'
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
