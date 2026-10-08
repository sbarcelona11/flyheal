import json
import hashlib
import numpy as np
import torch
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO
from flyheal.connectome import load_graph, matrices, rewire_edges, ConnectomeExtractor
from flyheal.models import FlyExtractor
from flyheal.environments import make_env


def graph_fixture(tmp_path):
    nodes=[dict(id=i,type=f'cell{i}',position=[i,i,0],nt='acetylcholine',sign=0 if i==7 else (1 if i%2 else -1),role='interneuron') for i in range(80)]
    edges=[[i,(i+j)%80,j+1] for i in range(80) for j in (1,2,3)]
    graph=dict(nodes=nodes,edges=edges,inputs=[[i,c] for c in range(8) for i in range(c*4,c*4+4)],outputs=list(range(64,80)),channels=[f'c{i}' for i in range(8)])
    path=tmp_path/'graph.json'; path.write_text(json.dumps(graph))
    validation=dict(status='passed',graphSha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks={'test':True})
    (tmp_path/'validation.json').write_text(json.dumps(validation))
    return path


def test_asset_gate_and_contact_orientation(tmp_path):
    path=graph_fixture(tmp_path)
    graph=load_graph(path)
    counts, inputs=matrices(graph)
    assert counts[1,0]==2 and counts[0,1]==0
    assert counts.sum()==720
    assert inputs.shape==(80,8) and inputs.sum()==32
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='hash'):
        load_graph(path)


def test_degree_preserving_rewire_and_source_contacts(tmp_path):
    graph=load_graph(graph_fixture(tmp_path))
    a=np.array(graph['edges']); b=np.array(rewire_edges(graph['edges'],80,3))
    assert not np.array_equal(a,b)
    for axis in (0,1):
        np.testing.assert_array_equal(np.bincount(a[:,axis],minlength=80),np.bincount(b[:,axis],minlength=80))
    for source in range(80):
        np.testing.assert_array_equal(np.sort(a[a[:,0]==source,2]),np.sort(b[b[:,0]==source,2]))
    assert len({tuple(e[:2]) for e in b})==len(b)


def test_input_fixed_topology_positive_gains_and_all_iteration_silence(tmp_path):
    path=graph_fixture(tmp_path)
    ext=ConnectomeExtractor(spaces.Box(-1,1,(8,),dtype=np.float32),kind='male_cns',width=80,graph_path=str(path))
    observations=torch.ones(4,8)
    drive, final=ext.activations(observations)
    assert drive.shape==final.shape==(4,80)
    assert not torch.count_nonzero(drive[:,32:])
    assert torch.all(ext.edge_gains()>0)
    ext.lesion('neurons',.5,5)
    optimizer=torch.optim.Adam(ext.parameters(),lr=.01)
    for _ in range(3):
        optimizer.zero_grad(); ext(observations).sum().backward(); optimizer.step()
        for activity in ext.iteration_activities(observations):
            assert not torch.count_nonzero(activity[:,ext.alive==0])
    effective=ext.weight_matrix()
    assert not torch.count_nonzero(effective[:,ext.alive==0])
    assert not torch.count_nonzero(effective[ext.alive==0,:])
    # Source 7 has unknown sign even though its measured anatomical edges exist.
    assert ext.original_mask[:,7].sum()>0 and not torch.count_nonzero(ext.base_signed[:,7])
    assert not torch.count_nonzero(effective[ext.original_mask==0])


def test_ppo_serialization_and_synthetic_activation_compatibility(tmp_path):
    path=graph_fixture(tmp_path)
    env=make_env('foraging')
    model=PPO('MlpPolicy',env,n_steps=64,batch_size=32,n_epochs=1,seed=0,
              policy_kwargs={'features_extractor_class':ConnectomeExtractor,
                             'features_extractor_kwargs':dict(kind='male_cns',width=80,graph_path=str(path)),
                             'net_arch':[]})
    model.learn(128)
    ext=model.policy.features_extractor; ext.lesion('edges',.5,9); ext.lesion('neurons',.5,9)
    model.learn(64)
    model.save(tmp_path/'policy')
    loaded=PPO.load(tmp_path/'policy')
    for name in ['mask','alive','base_counts','input_map','original_mask']:
        np.testing.assert_array_equal(getattr(ext,name).numpy(),getattr(loaded.policy.features_extractor,name).numpy())
    obs,_=env.reset(seed=9)
    assert int(loaded.predict(obs)[0]) in (0,1,2)
    synthetic=FlyExtractor(env.observation_space,width=80)
    h1,h2=synthetic.activations(torch.zeros(1,8))
    assert h1.shape==h2.shape==(1,80)


def test_fixed_width_and_config_validation():
    from flyheal.training import training_config
    cfg=training_config(dict(architecture='male_cns',environment='foraging',width=80))
    assert cfg['width']==80 and cfg['architectures']==['male_cns']
    assert cfg['environment']=='foraging'
    with pytest.raises(ValueError):
        training_config(dict(architecture='male_cns',environment='foraging',width=64))


def test_validated_actual_malecns_subset_and_rewired_model():
    graph=load_graph()
    assert len(graph['nodes'])==80 and len(graph['edges'])==1296
    assert sum(e[2] for e in graph['edges'])==26029
    space=make_env('foraging').observation_space
    measured=ConnectomeExtractor(space,kind='male_cns',width=80,seed=4)
    rewired=ConnectomeExtractor(space,kind='rewired_male_cns',width=80,seed=4)
    assert measured.effective_connections()==rewired.effective_connections()==1296
    assert not torch.equal(measured.mask,rewired.mask)
    for axis in (0,1):
        np.testing.assert_equal(measured.mask.sum(axis).numpy(),rewired.mask.sum(axis).numpy())
    np.testing.assert_equal(measured.input_map.numpy(),rewired.input_map.numpy())
    assert measured.modeled_connections()<measured.effective_connections()
    obs=torch.randn(3,8)
    first=measured(obs).detach()
    measured(torch.ones(4,8))
    torch.testing.assert_close(first,measured(obs))
    assert measured.graph_sha256==rewired.graph_sha256


def test_real_experiment_reports_data_and_model_assumptions(tmp_path):
    from pathlib import Path
    from flyheal.experiment import run
    from flyheal.report import build_report
    from flyheal.plots import plot_results
    config=json.loads(Path('configs/smoke.json').read_text())
    config.update(environment='foraging',architectures=['male_cns'],width=80,eval_episodes=1)
    run(config,tmp_path)
    report=build_report(tmp_path)
    plot_results(tmp_path)
    metadata=json.loads((tmp_path/'metadata.json').read_text())
    assert metadata['connectome']['validation']['status']=='passed'
    assert metadata['connectome']['measured_directed_edges']==1296
    assert metadata['architectures'][0]['modeled_connections']==1272
    assert metadata['architectures'][0]['stored_parameters']==1708
    assert 'Sin datos reales de MaleCNS.' not in report
    assert 'contactos anatómicos reales' in report
    assert 'No son aferencias olfativas medidas' in report
    assert 'cinco iteraciones tanh' in report


def test_gray_atlas_is_hashed_real_soma_context_excluding_modeled_cells():
    import hashlib,json
    from flyheal.connectome import GRAPH_PATH,load_graph
    data=GRAPH_PATH.parent
    raw=(data/'atlas.json').read_bytes();atlas=json.loads(raw);manifest=json.loads((data/'atlas_manifest.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==manifest['sha256']
    assert len(atlas['points'])==atlas['count']==manifest['count']==124209
    ids=[p[0] for p in atlas['points']]
    assert len(set(ids))==len(ids)
    assert not set(ids).intersection(n['id'] for n in load_graph()['nodes'])
    assert all(len(p)==4 and all(isinstance(v,int) for v in p) for p in atlas['points'])
    assert atlas['source']['sha256']=='2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'
