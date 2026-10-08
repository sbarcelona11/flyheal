# FlyHeal — mosca en un mundo 2D

<img width="1280" height="942" alt="ui_preview" src="https://github.com/user-attachments/assets/fec6cacb-b27b-4b4e-b9e8-bf0a9ec6776f" />

Trabajo final de Inteligencia Artificial II. Integrantes: Mathias Rodriguez Agro y Sebastian Barcelona.

Agente PPO que busca alimento con ocho señales sensoriales limitadas y tres acciones: girar a izquierda, avanzar o girar a derecha. Todas producen movimiento. El entorno contiene tres alimentos, cinco obstáculos y un límite de 500 pasos. Se evalúan lesiones permanentes y reentrenamiento de los parámetros supervivientes.

**Esta versión contiene únicamente el mundo 2D. Los entrenamientos anteriores fueron eliminados.** La web abre sin modelos guardados; inicia una nueva corrida para ver comportamiento, actividad y decisiones reales. Las pruebas usan modelos temporales y no son resultados experimentales del trabajo.

## Instalación e inicio

Python 3.10 o superior, CPU. Desde esta carpeta:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m flyheal.ui
```

Abrir http://127.0.0.1:8765/. Elegir arquitectura, presupuesto y semillas en **Entrenar una nueva red**. Al finalizar se habilitan mundo 2D, cerebro, decisiones, gráficos y métricas. Puede cancelarse el entrenamiento; solo se admite un trabajo activo. Las corridas completas se guardan en `results/web/` y se descubren al reiniciar. Sin modelo no se presentan decisiones aprendidas ficticias.

## Docker y Docker Compose

Requiere Docker en ejecución y Docker Compose. La imagen usa Python 3.12,
las versiones de `requirements-lock.txt` y PyTorch para CPU; no requiere GPU
ni instalar Python en el equipo. La primera construcción descarga las dependencias.

Desde esta carpeta:

```sh
docker compose up --build -d
```

Abrir http://127.0.0.1:8765/. El servicio publica el puerto solo en el equipo
local. Las corridas y checkpoints se guardan en `./results`, montado en
`/app/results`; persisten al detener o recrear el contenedor.

```sh
docker compose logs -f flyheal
docker compose down
```

Si el puerto está ocupado, usar por ejemplo `FLYHEAL_PORT=8876 docker compose up --build -d`
y abrir http://127.0.0.1:8876/. Se cambia también el puerto interno para conservar
la validación de origen de las solicitudes de entrenamiento.

Si se usa el ejecutable independiente, los comandos equivalentes son
`docker-compose up --build -d` y `docker-compose down`.
En Linux, los archivos creados por el contenedor pueden pertenecer a root.
Para usar el usuario local con Compose:

```sh
mkdir -p results
docker compose run --rm --service-ports --user "$(id -u):$(id -g)" flyheal
```

También se puede ejecutar con Docker directamente:

```sh
docker build -t flyheal:local .
mkdir -p results
docker run --rm --init --name flyheal \
  -p 127.0.0.1:8765:8765 \
  --mount "type=bind,source=$(pwd)/results,target=/app/results" \
  flyheal:local
```

Para ejecutar la comparación desde la terminal usando la misma imagen
(elegir una carpeta de salida nueva):

```sh
docker compose run --rm flyheal python -m flyheal.experiment --config configs/malecns_quick.json --out results/docker_run
docker compose run --rm flyheal python -m flyheal.plots --results results/docker_run
docker compose run --rm flyheal python -m flyheal.report --results results/docker_run
```

La imagen incluye las pruebas Python: `docker compose run --rm flyheal python -m pytest -q`.
Las pruebas JavaScript requieren Node en el equipo, como se indica más abajo.
Después de modificar el código, volver a ejecutar `docker compose up --build -d`.
El código está incluido en la imagen y `results/` se excluye de la construcción.

## Arquitecturas y datos

- **MaleCNS medido:** subgrafo validado de 80 neuronas, 1.296 conexiones dirigidas y 26.029 contactos sinápticos. Tamaño y topología fijos, ganancias entrenables, cinco iteraciones de dinámica artificial y lector PPO 80→3.
- **MaleCNS recableado:** conserva grados dirigidos y contactos por origen, cambiando destinos. No conserva contactos entrantes por destino.
- **Densa, estructurada sintética y dispersa aleatoria:** dos capas con 64/80/128/256 unidades por capa. Conectividad configurable en las sintéticas; el control aleatorio conserva los grados del estructurado.

La comparación principal utiliza MaleCNS, recableado y densa de ancho 80. La densa tiene otra capacidad y dinámica: no es una comparación equiparada en parámetros.

Los datos anatómicos son reales; sensores, dinámica y conducta son artificiales. El subconjunto visual→descendente no es un circuito olfativo validado. El atlas gris muestra somas sin actividad modelada; solo 80 células muestran actividad calculada. Las líneas son aristas, no neuritas. Más neuronas no garantizan mejor desempeño.

[Procedencia MaleCNS](docs/procedencia_malecns.md) y archivos de validación en `flyheal/data/malecns80/`. Código MIT; datos CC BY 4.0. Atribución en [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Comparación reproducible desde la terminal

```sh
python -m flyheal.experiment --config configs/malecns_quick.json --out results/new_run
python -m flyheal.plots --results results/new_run
python -m flyheal.report --results results/new_run
python -m flyheal.ui --results results/new_run
```

Tres arquitecturas y tres semillas en CPU. Usar una carpeta nueva por corrida. `configs/foraging_quick.json` define la comparación sintética; `configs/smoke.json` prueba funcionamiento, sin presupuesto suficiente para demostrar aprendizaje.

Exportar un episodio real de los checkpoints:

```sh
python -m flyheal.demo --results results/new_run --architecture male_cns --headless --gif results/new_run/demo.gif
```

## Protocolo y límites

Entrenamiento inicial y copia del checkpoint en tres ramas: continuación sana, lesión del 50% de unidades y lesión del 50% de conexiones. Daño persistente y reinicio simétrico de Adam. Evaluación determinista con mapas reservados emparejados. Se registran retorno, alimentos, colisiones, éxito, caída inmediata, ganancia y brecha con control sano. La fracción recuperada solo corresponde si hubo una pérdida; retornos negativos no justifican el umbral del 90%.

Primero comprobar que la política sana recolecta alimento. Girar sin chocar puede obtener mejor retorno que avanzar contra obstáculos; el retorno no basta para demostrar éxito. La configuración rápida no garantiza convergencia. Esta limpieza no modifica recompensa ni entrenamiento y no resuelve por sí sola esa conducta.

Cada corrida genera CSV, tablas agregadas, metadatos, máscaras, checkpoints, gráficos e informe. La demo de un episodio no sustituye la evaluación agregada.

## Pruebas

```sh
pip install pytest
python -m pytest -q
node --test tests/test_brain.mjs tests/test_decision.mjs tests/test_playback.mjs
```

Entorno reproducible, inferencia, lesiones persistentes, decisiones, procedencia, entrenamiento web, cancelación y arranque sin corridas.

## Documentos

[Propuesta](docs/propuesta_inicial.md) · [Informe base](docs/informe_tecnico.md) · [Guion de ocho minutos](docs/guion_8_minutos.md).

Completar los resultados finales después de entrenar nuevamente. La propuesta Word fechada describe el desarrollo previo; sus observaciones preliminares no son resultados incluidos en esta versión limpia.

Inspiración visual: [FlyDino / Mert Cobanov](https://github.com/cobanov/flyjump). Implementación propia que muestra actividad y acciones de las políticas de FlyHeal.
