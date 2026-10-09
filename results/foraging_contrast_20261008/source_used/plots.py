"""Figures: each independent replicate is one training seed."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

LABELS = {'male_cns': 'MaleCNS medido (80 células)', 'rewired_male_cns': 'MaleCNS reordenado', 'dense': 'Densa', 'structured': 'Estructurada sintética', 'random': 'Aleatoria (grados iguales)'}
COLORS = {'male_cns': '#12805c', 'rewired_male_cns': '#bd651f', 'dense': '#2463a7', 'structured': '#12805c', 'random': '#bd651f'}


def plot_results(output):
    output = Path(output)
    dest = output / 'figures'
    dest.mkdir(exist_ok=True)
    measured = json.loads((output / 'metadata.json').read_text())['config']['architectures']
    measured = any(arch in ('male_cns', 'rewired_male_cns') for arch in measured)
    foraging = json.loads((output / 'metadata.json').read_text())['config'].get('environment', 'foraging') == 'foraging'
    limits = {} if foraging else {'ylim': (0, 525)}
    raw = pd.read_csv(output / 'episodes.csv')
    summary = pd.read_csv(output / 'summary.csv')
    seed_means = raw.groupby(['architecture', 'seed', 'condition', 'steps']).reward.mean().reset_index()
    conditions = list(summary.condition.unique())
    fig, axs = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4.3), squeeze=False)
    for ax, condition in zip(axs[0], conditions):
        for arch in raw.architecture.unique():
            for name, style in [(condition, '-'), ('healthy', '--')]:
                data = seed_means[(seed_means.architecture == arch) & (seed_means.condition == name)]
                agg = data.groupby('steps').reward.agg(['mean', 'sem'])
                x, y, error = agg.index.to_numpy(), agg['mean'].to_numpy(), agg['sem'].fillna(0).to_numpy()
                label = LABELS[arch] + (' · sano' if name == 'healthy' else '')
                ax.plot(x, y, style, color=COLORS[arch], label=label)
                if name == condition:
                    ax.fill_between(x, y-error, y+error, color=COLORS[arch], alpha=0.15)
        ax.set(title=condition.replace('neurons', 'Unidades').replace('edges', 'Conexiones'),
               xlabel='Pasos adicionales después de intervención', ylabel='Retorno medio' if foraging else 'Retorno medio (máximo 500)', **limits)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('FlyHeal · recuperación; bandas = ±1 error estándar entre semillas', fontsize=12)
    fig.tight_layout()
    fig.savefig(dest / 'recovery.png', dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4.3), squeeze=False)
    for ax, condition in zip(axs[0], conditions):
        for i, arch in enumerate(raw.architecture.unique()):
            data = summary[(summary.condition == condition) & (summary.architecture == arch)]
            columns = ['pre', 'post', 'recovered', 'healthy']
            mean = data[columns].mean().to_numpy()
            err = data[columns].sem().fillna(0).to_numpy()
            x = np.arange(4) + (i - 1) * .22
            ax.bar(x, mean, width=.22, yerr=err, capsize=3, color=COLORS[arch], label=LABELS[arch])
            for row in data[columns].to_numpy():
                ax.scatter(x, row, color='black', s=8, alpha=.5, zorder=3)
        ax.set_xticks(range(4), ['Antes', 'Daño', 'Recuperado', 'Sano + pasos'])
        ax.set(title=condition, ylabel='Retorno', **limits)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Etapas · puntos = semillas; barras de error = ±1 error estándar')
    fig.tight_layout()
    fig.savefig(dest / 'stages.png', dpi=160)
    plt.close(fig)

    architectures = list(raw.architecture.unique())
    fig, axs = plt.subplots(1, len(architectures), figsize=(4 * len(architectures), 4), squeeze=False)
    seed = int(raw.seed.min())
    for ax, arch in zip(axs[0], architectures):
        model = PPO.load(output / 'checkpoints' / f'{arch}_seed{seed}' / 'baseline.zip', device='cpu')
        mask = model.policy.features_extractor.mask.cpu().numpy()
        ax.imshow(mask, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
        ax.set(title=f'{LABELS[arch]}\n{int(mask.sum())} aristas', xlabel='Origen', ylabel='Destino')
    fig.suptitle(f'Topologías prelesión · semilla {seed}' if measured else f'Conectividad sintética prelesión · semilla {seed}')
    fig.tight_layout()
    fig.savefig(dest / 'connectivity.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arch in architectures:
        data = seed_means[(seed_means.architecture == arch) & (seed_means.condition == 'baseline')]
        agg = data.groupby('steps').reward.agg(['mean', 'sem'])
        x, y, err = agg.index.to_numpy(), agg['mean'].to_numpy(), agg['sem'].fillna(0).to_numpy()
        ax.plot(x, y, color=COLORS[arch], label=LABELS[arch])
        ax.fill_between(x, y-err, y+err, color=COLORS[arch], alpha=.15)
    ax.set(title='Aprendizaje inicial · media ±1 error estándar', xlabel='Pasos de entrenamiento', ylabel='Retorno', **limits)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(dest / 'baseline.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results/new_run'))
    plot_results(parser.parse_args().results)


if __name__ == '__main__':
    main()
