import json
from pathlib import Path
from flyheal.experiment import run
from flyheal.report import build_report


def test_report_records_actual_budget_and_only_valid_recovery_ratios(tmp_path):
    cfg = json.loads(Path('configs/smoke.json').read_text())
    cfg['fractions'] = [0.0]
    run(cfg, tmp_path)
    report = build_report(tmp_path)
    assert '2.048' in report  # 512 baseline + 3 branches of 512
    assert 'N/A' in report   # no immediate drop after zero lesion
    assert (tmp_path / 'report.md').read_text() == report
