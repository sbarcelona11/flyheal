import json
from pathlib import Path
from PIL import Image
from flyheal.experiment import run
from flyheal.plots import plot_results
from flyheal.demo import play


def test_plots_and_headless_demo_use_real_saved_run(tmp_path):
    cfg = json.loads(Path('configs/smoke.json').read_text())
    run(cfg, tmp_path)
    plot_results(tmp_path)
    for name in ['recovery.png', 'stages.png', 'connectivity.png']:
        image = Image.open(tmp_path / 'figures' / name)
        assert image.width > 500 and image.height > 300
    returns = play(tmp_path, 'structured', 0, 'neurons_0.5', 'all',
                   episodes=1, headless=True, gif=tmp_path / 'demo.gif')
    assert set(returns) == {'baseline', 'post', 'recovered'}
    assert all(isinstance(value[0], float) for value in returns.values())
    gif = Image.open(tmp_path / 'demo.gif')
    assert gif.n_frames > 1
