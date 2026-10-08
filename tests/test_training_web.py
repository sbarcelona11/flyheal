import time
import json
from pathlib import Path
import numpy as np
import pytest
from flyheal.models import connectivity
from flyheal.training import TrainingManager, training_config


def test_density_is_configurable_and_random_control_preserves_both_degrees():
    a=connectivity('structured', 128, 3, density=.5)
    b=connectivity('random', 128, 3, density=.5)
    assert a.sum()==8192
    assert np.all(a.sum(0)==64) and np.all(a.sum(1)==64)
    assert np.array_equal(a.sum(0),b.sum(0)) and np.array_equal(a.sum(1),b.sum(1))
    assert not np.array_equal(a,b)
    assert np.array_equal(connectivity('structured',64,0),connectivity('structured',64,0,density=.3125))
    with pytest.raises(ValueError):connectivity('structured',64,0,density=0)


def test_web_config_limits_compute_and_records_requested_shape():
    cfg=training_config({'architecture':'structured','width':128,'density':.5,'budget':'fast','seeds':1})
    assert cfg['width']==128 and cfg['density']==.5 and cfg['seeds']==[0]
    assert cfg['baseline_steps']==20480 and cfg['recovery_steps']==10240
    for options in [{'width':4096}, {'density':2}, {'seeds':100}, {'budget':'unknown'}]:
        with pytest.raises(ValueError):training_config(options)


def test_background_worker_finishes_and_is_discovered_after_restart(tmp_path):
    manager=TrainingManager(Path('results/new_run'),training_root=tmp_path)
    cfg=json.loads(Path('configs/smoke.json').read_text());cfg['width']=128;cfg['density']=.5
    job=manager.start_config(cfg)
    try:
        deadline=time.monotonic()+25
        while time.monotonic()<deadline:
            state=manager.status()
            if state['status'] in ['complete','failed']:break
            time.sleep(.1)
        assert state['status']=='complete',state
        assert state['progress']['trained_steps']==2048
        assert state['progress']['percent']==100
        run=next(r for r in manager.runs()['runs'] if r['id']==job['id'])
        assert run['width']==128
        store=manager.store(job['id'])
        assert store.config['density']==.5
        assert len(store.replay('structured',0,'neurons_0.5','post')['alive'])==128
        reopened=TrainingManager(Path('results/new_run'),training_root=tmp_path)
        assert any(r['id']==job['id'] for r in reopened.runs()['runs'])
    finally:manager.close()


def test_cancel_terminates_only_owned_training_job_and_keeps_files(tmp_path):
    manager=TrainingManager(Path('results/new_run'),training_root=tmp_path)
    job=manager.start({'architecture':'dense','width':256,'budget':'extended','density':1,'seeds':3})
    try:
        with pytest.raises(RuntimeError):manager.start({'architecture':'dense'})
        state=manager.cancel()
        assert state['status']=='cancelled'
        assert (tmp_path/job['id']/'config.json').is_file()
        assert manager.process.poll() is not None
    finally:manager.close()


def test_worker_exits_when_owning_server_disappears(tmp_path):
    import subprocess
    import sys
    owner_code = """
import sys, time
from flyheal.training import TrainingManager
manager = TrainingManager('results/new_run', sys.argv[1])
job = manager.start({'width':256,'budget':'extended','seeds':3})
print(job['id'], flush=True)
time.sleep(60)
"""
    owner=subprocess.Popen([sys.executable,'-u','-c',owner_code,str(tmp_path)],stdout=subprocess.PIPE,text=True)
    identifier=owner.stdout.readline().strip()
    try:
        deadline=time.monotonic()+10
        directory=tmp_path/identifier
        while not (directory/'progress.json').exists() and time.monotonic()<deadline:
            time.sleep(.1)
        assert (directory/'progress.json').exists()
        owner.kill();owner.wait(timeout=3)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            job=json.loads((directory/'job.json').read_text())
            if job['status']=='cancelled':break
            time.sleep(.1)
        assert job['status']=='cancelled'
    finally:
        if owner.poll() is None:owner.kill();owner.wait(timeout=3)
        # Clean up the old implementation's orphan if this regression fails.
        if 'job' in locals() and job.get('status')=='running':
            import os, signal
            if job.get('worker_pid'):os.kill(job['worker_pid'],signal.SIGTERM)
