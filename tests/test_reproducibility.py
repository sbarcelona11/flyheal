import hashlib
import json
from pathlib import Path
import pandas as pd
from flyheal.experiment import run


def test_same_config_repeats_raw_results_and_preserves_source_hashes(tmp_path):
    cfg = json.loads(Path('configs/smoke.json').read_text())
    first, second = tmp_path / 'first', tmp_path / 'second'
    run(cfg, first)
    run(cfg, second)
    pd.testing.assert_frame_equal(pd.read_csv(first / 'episodes.csv'),
                                  pd.read_csv(second / 'episodes.csv'))
    meta = json.loads((first / 'metadata.json').read_text())
    assert meta['total_training_steps'] == 2048
    for name, expected_hash in meta['source_sha256'].items():
        actual = hashlib.sha256((first / 'source_used' / name).read_bytes()).hexdigest()
        assert actual == expected_hash
