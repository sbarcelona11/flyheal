"""Generate Spanish result tables directly from saved experimental data."""
import argparse
import json
from pathlib import Path
import pandas as pd

NAMES = {'male_cns': 'MaleCNS medido', 'rewired_male_cns': 'MaleCNS reordenado', 'dense': 'Densa', 'structured': 'Estructurada sintética', 'random': 'Aleatoria'}


def number(value, decimals=1):
    return f'{value:.{decimals}f}' if pd.notna(value) else 'N/A'


def mean_sem(series, decimals=1):
    valid = series.dropna()
    if not len(valid):
        return 'N/A (n=0)'
    return f'{number(valid.mean(), decimals)} ± {number(valid.sem(), decimals)} (n={len(valid)})'


def build_report(results):
    results = Path(results)
    meta = json.loads((results / 'metadata.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError('Only completed runs may be reported')
    summary = pd.read_csv(results / 'summary.csv')
    cfg = meta['config']
    measured = any(arch in ('male_cns', 'rewired_male_cns') for arch in cfg['architectures'])
    foraging = cfg.get('environment', 'foraging') == 'foraging'
    total = f"{meta['total_training_steps']:,}".replace(',', '.')
    episodes = f"{meta['evaluation_episodes']:,}".replace(',', '.')
    text = f"""# Resultados reales — {results.name}

Ejecución finalizada: {meta['finished_utc']} (UTC). CPU, un hilo PyTorch.
Tiempo registrado: {meta['elapsed_seconds']:.1f} segundos ({meta['elapsed_seconds']/60:.2f} minutos),
sin incluir instalación, documentación ni pruebas. Total: **{total} pasos de
entrenamiento** y **{episodes} episodios de evaluación**.

Configuración: semillas {cfg['seeds']}; {cfg['baseline_steps']:,} pasos iniciales,
{cfg['recovery_steps']:,} por rama; {cfg['eval_episodes']} episodios por punto;
{cfg['ppo']['n_epochs']} épocas PPO. La misma política inicial se ramifica en
continuación sana y lesiones. Los checkpoints finales se fijan por presupuesto.

Media ±1 error estándar entre semillas. n es el número de semillas válidas;
no es un intervalo de confianza de 95%. Retornos medios sin descontar, máximo 500.

| Arquitectura | Lesión | Antes B | Daño D | Final R | Sano H |
|---|---|---|---|---|---|
"""
    for (arch, condition), group in summary.groupby(['architecture', 'condition']):
        values = ' | '.join(mean_sem(group[col]) for col in ['pre', 'post', 'recovered', 'healthy'])
        text += f'| {NAMES[arch]} | {condition} | {values} |\n'
    text += '\n| Arquitectura | Lesión | Caída B−D | Ganancia R−D | Fracción recuperada | Brecha H−R |\n|---|---|---|---|---|---|\n'
    for (arch, condition), group in summary.groupby(['architecture', 'condition']):
        values = ' | '.join(mean_sem(group[col], 3 if col == 'recovered_fraction' else 1)
                            for col in ['drop', 'gain', 'recovered_fraction', 'healthy_gap'])
        text += f'| {NAMES[arch]} | {condition} | {values} |\n'
    text += """
La fracción se calcula **por semilla y solo cuando B>D**; N/A significa que
no hubo pérdida que recuperar. Los conteos pueden diferir. Un valor mayor a 1
supera el nivel prelesión; valores elevados requieren revisar el denominador.
Una caída negativa indica una mejora inmediata por ablación. Un final R menor
que D indica deterioro durante el entrenamiento, aunque R pueda superar B.

## Valores por semilla

| Arquitectura | Semilla | Lesión | B | D | R | H | Primer checkpoint ≥90% de B |
|---|---|---|---:|---:|---:|---:|---:|
"""
    for row in summary.itertuples():
        text += (f'| {NAMES[row.architecture]} | {row.seed} | {row.condition} | '
                 f'{row.pre:.1f} | {row.post:.1f} | {row.recovered:.1f} | {row.healthy:.1f} | '
                 f'{number(row.steps_to_90pct if not foraging or row.pre > 0 else float("nan"), 0)} |\n')
    text += """
N/A en el checkpoint indica que el umbral no se alcanzó en el presupuesto;
no prueba que nunca pudiera alcanzarse. Alcanzarlo no exige mantenerlo. El
umbral es relativo a B, que puede ser negativo o corresponder a poca comida recolectada. El umbral del 90% se informa solo cuando B es positivo.

## Gráficos y trazabilidad

![Aprendizaje inicial](figures/baseline.png)

![Impacto y recuperación](figures/recovery.png)

![Etapas y semillas](figures/stages.png)

![Máscaras sintéticas](figures/connectivity.png)

Fuente de cifras: `episodes.csv`, `summary.csv`, `aggregate.csv`,
`metadata.json`. Versiones y hashes se registran en metadata; `source_used/`
conserva el código de cálculo ejecutado. Sin datos reales de MaleCNS.
"""
    if foraging:
        text = text.replace('Retornos medios sin descontar, máximo 500.', 'Retornos medios sin descontar: comida +10, avance hacia comida ×2, paso −0,01 y colisión −0,15. Horizonte 500 pasos; el retorno no tiene máximo 500.')
        text = text.replace('negativo o corresponder a poca comida recolectada. El umbral del 90% se informa solo cuando B es positivo.', 'negativo o corresponder a poca comida recolectada. El umbral del 90% se informa solo cuando B es positivo.')
        text = text.replace('N/A en el checkpoint indica que el umbral no se alcanzó en el presupuesto;', 'N/A en el checkpoint indica que B no es positivo o que el umbral no se alcanzó en el presupuesto;')
        task = pd.read_csv(results / 'task_metrics.csv')
        chosen = task[(task.stage.isin(['pre', 'post', 'recovered'])) | ((task.condition == 'healthy') & (task.steps == cfg['recovery_steps']))]
        text += '\n## Comida y colisiones: tarea 2D\n\nCada cifra resume primero los episodios de una semilla y después las semillas. Éxito = recoger las tres comidas antes de 500 pasos.\n\n| Arquitectura | Rama | Etapa | Comidas (de 3) | Colisiones por episodio | Éxito (fracción) |\n|---|---|---|---|---|---|\n'
        for (arch, condition, stage), group in chosen.groupby(['architecture', 'condition', 'stage']):
            values = ' | '.join(mean_sem(group[col], 2) for col in ['mean_foods', 'mean_collisions', 'success_rate'])
            text += f'| {NAMES[arch]} | {condition} | {stage} | {values} |\n'
        text += '\nLas observaciones son ocho sensores; los mapas completos pertenecen solo al visor. El mundo, las conexiones y las lesiones son sintéticos. Tres semillas y este presupuesto exploratorio no establecen superioridad de una arquitectura. Fuente adicional: `task_metrics.csv`.\n'
    if measured:
        text = text.replace('Sin datos reales de MaleCNS.', 'Con datos reales MaleCNS validados: ver procedencia y límites abajo.')
        text = text.replace('![Máscaras sintéticas]', '![Topologías medidas y controles]')
        text = text.replace('El mundo, las conexiones y las lesiones son sintéticos.', 'El mundo y las lesiones son simulados. MaleCNS usa contactos anatómicos reales; el control reordena sus aristas y la red densa es sintética.')
        circuit = meta['connectome']
        text += f'\n## Procedencia MaleCNS y supuestos\n\nSubconjunto inducido de **{circuit["cells"]} células**, **{circuit["measured_directed_edges"]} pares dirigidos** y **{circuit["contacts"]} contactos**; validación independiente frente a las tablas oficiales completada. SHA256 del grafo: `{circuit["graph_sha256"]}`.\n\n'
        text += 'Solo la topología MaleCNS corresponde al subgrafo medido. El control preserva los grados de entrada/salida y la distribución de contactos por fuente, pero reordena destinos; la línea densa controla capacidad con dos capas de ancho 80, no número idéntico de parámetros. Las diferencias incluyen dinámica y número de parámetros: esto no aísla exclusivamente el efecto de topología.\n\n'
        text += 'Los ocho sensores de búsqueda de alimento se inyectan por una asignación artificial a 32 células de ocho tipos visuales. No son aferencias olfativas medidas. El modelo usa cinco iteraciones tanh con fuga 0,5, estado reiniciado en cada decisión, y ganancias positivas aprendidas por conexión. Los signos según transmisor son supuestos: acetilcolina positiva, GABA/glutamato negativos; desconocido cero. Las aristas de signo desconocido permanecen en el conteo anatómico y no transmiten actividad modelada. PPO aprende actor/crítico a partir de las 80 actividades, no un controlador motor biológico validado.\n\n'
        text += '| Arquitectura | Aristas anatómicas/control | Aristas modeladas con signo | Parámetros almacenados de política |\n|---|---:|---:|---:|\n'
        for arch in cfg['architectures']:
            records = [row for row in meta['architectures'] if row['architecture']==arch]
            row = records[0]
            text += f'| {NAMES[arch]} | {row.get("anatomical_connections", row["active_layer2_connections"])} | {row.get("modeled_connections", row["active_layer2_connections"])} | {row["stored_parameters"]} |\n'
        text += '\nEn la red densa, la lesión de neuronas silencia unidades de la segunda capa. En los circuitos de 80 células, silencia fuentes y destinos en todas las iteraciones; estas intervenciones no son dinámicamente idénticas. La lesión de aristas selecciona pares anatómicos, incluidos los de signo desconocido. Ninguna lesión se revierte al reentrenar. El entrenamiento modifica ganancias y lectura, no crea células ni aristas. Validación y atribuciones: `metadata.json`, `flyheal/data/malecns80/validation.json`, `NOTICE.md` y `docs/procedencia_malecns.md`. No demuestra reparación de un cerebro real ni superioridad general.\n'
    (results / 'report.md').write_text(text, encoding='utf-8')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results/new_run'))
    build_report(parser.parse_args().results)


if __name__ == '__main__':
    main()
