import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from flyheal.ui import ReplayStore, make_server

RESULTS = None


@pytest.fixture(scope='module', autouse=True)
def fresh_results(tmp_path_factory):
    global RESULTS
    from flyheal.experiment import run
    RESULTS = tmp_path_factory.mktemp('ui_simulation')
    config = json.loads(Path('configs/smoke.json').read_text())
    config['seeds'] = [0, 1, 2]
    run(config, RESULTS)



def test_replay_is_real_saved_policy_and_preserves_silenced_units():
    store = ReplayStore(RESULTS)
    baseline = store.replay('structured', 0, 'neurons_0.5', 'baseline')
    damaged = store.replay('structured', 0, 'neurons_0.5', 'post')
    assert isinstance(baseline['return'], float)
    assert isinstance(damaged['return'], float)
    assert 2 <= len(damaged['frames']) <= 501
    assert damaged['frames'][0]['step'] == 0
    assert damaged['frames'][-1]['step'] == len(damaged['frames']) - 1
    assert damaged['alive'].count(0) == 32
    for frame in damaged['frames']:
        assert all(frame['h2'][i] == 0 for i, alive in enumerate(damaged['alive']) if alive == 0)
    model = PPO.load(RESULTS / 'checkpoints/structured_seed0/neurons_0.5_post.zip', device='cpu')
    frame = damaged['frames'][0]
    with torch.no_grad():
        expected = model.policy.features_extractor(torch.tensor([frame['state']], dtype=torch.float32)).numpy()[0]
    np.testing.assert_allclose(frame['h2'], expected, atol=2e-6)


def test_summary_has_real_means_and_invalid_selections_are_rejected():
    store = ReplayStore(RESULTS)
    summary = store.summary()
    row = next(x for x in summary['seeds'] if x['architecture'] == 'structured' and x['seed'] == 0 and x['condition'] == 'neurons_0.5')
    import pandas as pd
    raw = pd.read_csv(RESULTS / 'episodes.csv')
    pre = raw[(raw.architecture == 'structured') & (raw.seed == 0) & (raw.condition == 'baseline') & (raw.steps == store.config['baseline_steps'])].reward.mean()
    assert row['pre'] == pytest.approx(pre)
    assert summary['provenance'] == 'synthetic'
    with pytest.raises(ValueError):
        store.replay('../../etc', 0, 'neurons_0.5', 'post')
    with pytest.raises(ValueError):
        store.replay('structured', 99, 'neurons_0.5', 'post')
    with pytest.raises(ValueError):
        store.replay('structured', 0, 'neurons_0.5', 'unknown')


def test_local_http_serves_ui_json_and_rejects_unsafe_paths():
    server = make_server(RESULTS, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(address) as response:
            assert response.status == 200
            assert 'FlyHeal' in response.read().decode()
        with urllib.request.urlopen(address + '/api/summary') as response:
            assert json.load(response)['config']['seeds'] == [0, 1, 2]
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(address + '/api/replay?architecture=../../etc&seed=0&condition=neurons_0.5&stage=post')
        assert error.value.code == 400
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(address + '/../../requirements-lock.txt')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cached_baseline_keeps_requested_intervention_metadata():
    store = ReplayStore(RESULTS)
    first = store.replay('structured', 0, 'neurons_0.5', 'baseline')
    second = store.replay('structured', 0, 'edges_0.5', 'baseline')
    assert first['condition'] == 'neurons_0.5'
    assert second['condition'] == 'edges_0.5'
    assert first['frames'] == second['frames']


def test_training_http_rejects_foreign_origin_and_accepts_local_configuration(tmp_path):
    server = make_server(RESULTS, port=0, training_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    address = f'http://127.0.0.1:{server.server_port}'
    body = json.dumps({'architecture':'structured','width':128,'density':.5,'budget':'fast','seeds':1}).encode()
    try:
        request=urllib.request.Request(address+'/api/train',data=body,headers={'Content-Type':'application/json','Origin':'https://foreign.example'})
        with pytest.raises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
        assert error.value.code==403
        request=urllib.request.Request(address+'/api/train',data=body,headers={'Content-Type':'application/json','Origin':address})
        with urllib.request.urlopen(request) as response:
            job=json.load(response);assert response.status==202
        assert job['config']['width']==128
        with urllib.request.urlopen(address+'/api/progress') as response:assert json.load(response)['status']=='running'
        request=urllib.request.Request(address+'/api/cancel',data=b'{}',headers={'Content-Type':'application/json','Origin':address})
        with urllib.request.urlopen(request) as response:assert json.load(response)['status']=='cancelled'
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


def test_brain_replay_exposes_original_edges_to_distinguish_removed_connections():
    store = ReplayStore(RESULTS)
    baseline = store.replay('structured', 0, 'edges_0.5', 'baseline')
    damaged = store.replay('structured', 0, 'edges_0.5', 'post')
    recovered = store.replay('structured', 0, 'edges_0.5', 'recovered')
    original = np.array(damaged['original_mask'])
    assert np.array_equal(original, baseline['mask'])
    assert np.array_equal(original, recovered['original_mask'])
    assert int(original.sum()) == 1280
    assert int(original.sum()-np.array(damaged['mask']).sum()) == 640


def test_foraging_replay_uses_step_counter_and_signed_reward(tmp_path):
    from flyheal.experiment import run
    config=json.loads(Path('configs/smoke.json').read_text())
    config['environment']='foraging'
    run(config,tmp_path)
    replay=ReplayStore(tmp_path).replay('structured',0,'neurons_0.5','baseline')
    assert replay['environment']=='foraging'
    assert len(replay['frames'][0]['state'])==8
    assert replay['frames'][0]['world']['step']==0
    assert replay['frames'][-1]['step']==len(replay['frames'])-1
    assert replay['frames'][-1]['world']['step']==replay['frames'][-1]['step']
    assert isinstance(replay['return'],float)
    assert 'foods_collected' in replay['task_metrics']


def test_malecns_replay_has_verified_ids_single_population_and_actual_coordinates(tmp_path):
    from flyheal.experiment import run
    config=json.loads(Path('configs/smoke.json').read_text())
    config.update(environment='foraging',architectures=['male_cns'],width=80)
    run(config,tmp_path)
    replay=ReplayStore(tmp_path).replay('male_cns',0,'neurons_0.5','post')
    assert replay['population']=='single'
    assert replay['provenance']=='malecns-measured-subgraph'
    assert len(replay['cells'])==80 and replay['alive'].count(0)==40
    assert replay['cells'][0]['id']==10010
    assert replay['cells'][0]['position']==[58238,22234,36276]
    assert len(replay['frames'][0]['h2'])==80
    assert all(frame['h2'][i]==0 for frame in replay['frames'] for i,alive in enumerate(replay['alive']) if not alive)


def test_decision_readout_exposes_exact_logits_and_probabilities():
    replay=ReplayStore(RESULTS).replay('structured',0,'neurons_0.5','baseline')
    frame=replay['frames'][0]
    weights=np.asarray(replay['decision']['weights'])
    bias=np.asarray(replay['decision']['bias'])
    np.testing.assert_allclose(weights @ np.asarray(frame['h2']) + bias,frame['decision']['logits'],atol=3e-6)
    logits=np.asarray(frame['decision']['logits']);expected=np.exp(logits-logits.max());expected/=expected.sum()
    np.testing.assert_allclose(expected,frame['decision']['probabilities'],atol=2e-6)
    assert int(np.argmax(logits))==frame['action']
    assert replay['decision']['architecture']=='linear'
