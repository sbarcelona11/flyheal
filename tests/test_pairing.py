import json
from pathlib import Path
import pandas as pd
import torch
from flyheal.experiment import run


def test_zero_damage_branches_have_identical_training_and_do_not_overwrite(tmp_path):
    config = json.loads(Path('configs/smoke.json').read_text())
    config['fractions'] = [0.0]
    run(config, tmp_path)
    raw = pd.read_csv(tmp_path / 'episodes.csv')
    curves = {}
    for condition in ['healthy', 'neurons_0', 'edges_0']:
        curves[condition] = raw[raw.condition == condition].reward.tolist()
    assert curves['healthy'] == curves['neurons_0'] == curves['edges_0']
    import pytest
    with pytest.raises(FileExistsError):
        run(config, tmp_path)
