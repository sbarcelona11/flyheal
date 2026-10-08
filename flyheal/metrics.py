"""Metrics computed per training seed before aggregation."""
import numpy as np
import pandas as pd


def recovery_metrics(pre, post, recovered, healthy):
    drop = float(pre - post)
    gain = float(recovered - post)
    return {'drop': drop, 'gain': gain,
            'recovered_fraction': gain / drop if drop > 1e-9 else None,
            'retention': float(post / pre) if pre > 0 else None,
            'healthy_gap': float(healthy - recovered)}


def summarize(episodes: pd.DataFrame) -> pd.DataFrame:
    means = episodes.groupby(['architecture', 'seed', 'condition', 'steps']).reward.mean()
    rows = []
    for (arch, seed), group in episodes.groupby(['architecture', 'seed']):
        baseline = means.loc[arch, seed, 'baseline']
        pre = baseline.loc[baseline.index.max()]
        healthy = means.loc[arch, seed, 'healthy']
        for condition in group.condition.unique():
            if condition in ('baseline', 'healthy'):
                continue
            curve = means.loc[arch, seed, condition]
            end = curve.index.max()
            post, recovered = curve.loc[0], curve.loc[end]
            crossing = curve[curve >= 0.9 * pre] if pre > 0 else curve.iloc[:0]
            # Time is an observed checkpoint, not an exact or sustained crossing.
            rows.append({'architecture': arch, 'seed': seed, 'condition': condition,
                         'pre': pre, 'post': post, 'recovered': recovered,
                         'healthy': healthy.loc[end], 'recovery_steps': end,
                         'steps_to_90pct': int(crossing.index.min()) if len(crossing) else None,
                         **recovery_metrics(pre, post, recovered, healthy.loc[end])})
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = ['pre', 'post', 'recovered', 'healthy', 'drop', 'gain',
               'recovered_fraction', 'retention', 'healthy_gap']
    result = summary.groupby(['architecture', 'condition'])[metrics].agg(['mean', 'sem', 'count'])
    result.columns = ['_'.join(col) for col in result.columns]
    return result.reset_index()
