# Mundo 2D de búsqueda de alimento

`ForagingEnv` es un entorno Gymnasium continuo de 10 × 10 unidades, con tres
comidas, cinco obstáculos rectangulares y una mosca de radio 0,15. Cada reset
crea posiciones deterministas a partir de su semilla; evaluación y entrenamiento
usan entornos separados. Las semillas de evaluación son 100000 + 100 × semilla
de entrenamiento + episodio, compartidas por arquitectura y rama. No es un
modelo de comportamiento biológico validado. La comparación principal usa
conectividad real MaleCNS como restricción del controlador; el mundo, los
sensores y la tarea siguen siendo artificiales.

La política PPO recibe **ocho números**, nunca el mapa: intensidad de olor en
antenas izquierda/frontal/derecha; distancia libre mediante tres rayos de alcance
4; seno y coseno de orientación. Cada antena se sitúa 0,45 unidades adelante,
con ángulos ±0,6 radianes. Olor = max(0, 1 − distancia a comida / 4), sin señal
fuera del alcance. Olor y distancias están en [0,1]; orientación en [−1,1]. El
olor no simula difusión ni oclusión por paredes; es un campo local sencillo.

Acciones: **0 izquierda**, **1 avanzar**, **2 derecha**. Cada acción mueve 0,18
unidades; girar cambia primero la orientación ±0,35 radianes. La colisión se verifica en la posición propuesta al final de cada paso, no
mediante un barrido continuo del trayecto. Esta aproximación discreta puede
permitir roces de esquinas entre posiciones; no representa física de contacto
continua. Una posición final que choca con el límite o con un obstáculo conserva la posición y cuenta una
colisión. Puede cambiar la orientación para salir del choque. Los obstáculos
se separan y no bloquean la posición inicial. La comida no se repone. Se recoge
cuando la distancia entre centros es ≤0,45. Termina al recoger las tres comidas,
o se trunca a los 500 pasos.

Recompensa por paso: **10 por comida + 2 × (distancia anterior − nueva distancia)
− 0,01 − 0,15 si hay colisión**. El término de avance usa la comida restante más
cercana antes de recoger; la eliminación no produce un salto artificial en ese
paso. Es una ayuda densa de entrenamiento calculada por el entorno: la política
no recibe esa distancia. El olor es parcial, pero la recompensa de avance usa
información global del simulador. Este diseño permite aprender con presupuestos
CPU modestos; no equivale a entrenamiento puramente basado en éxito ni a un
modelo biológico. El retorno combina comida, progreso y colisiones, y puede ser
negativo. Se publican comida, colisiones y éxito por separado.

`env.snapshot()` devuelve un objeto JSON con dimensiones, mosca, comida,
obstáculos, ocho sensores y contadores para el visor; esta información **no** se
entrega al extractor de la política. La mosca usa coordenadas continuas en
`fly.x`, `fly.y`, `fly.heading`; alimentos usan `collected`; los obstáculos usan
`x`, `y`, `width`, `height`. Son datos de visualización del mundo completo.

## Comparación principal con conectividad real

`configs/malecns_quick.json`: `male_cns`, `rewired_male_cns` y densa de ancho 80.
El archivo MaleCNS conserva 80 neuronas, 1.296 conexiones dirigidas y 26.029
contactos sinápticos, verificados contra las tablas oficiales completas. La
procedencia, licencia CC BY 4.0 y comprobaciones están documentadas en
`docs/procedencia_malecns.md` y `flyheal/data/malecns80/`.

La selección original une células visuales con neuronas descendentes. No es un
circuito olfatorio validado. Inyectamos los ocho sensores en 32 células según
un mapa artificial: los nombres de tipos visuales no describen receptores de
olor. El circuito usa contactos normalizados, signos asumidos según
neurotransmisor, ganancias entrenables y cinco iteraciones tanh con fuga por
decisión. Su estado se reinicia en cada observación, sin memoria entre pasos.
PPO aprende una lectura actor-crítico de las 80 actividades. Los 16 índices de
salida originales se conservan como metadatos, no restringen la lectura.

El control recableado conserva grados dirigidos de entrada y salida, número
de conexiones y multiconjunto de contactos por origen. No conserva contactos
entrantes por destino ni la topología original. Mantiene etiquetas, inyección
y dinámica. La densa 8→80→80 tiene dos capas y distinta capacidad/dinámica;
no es un control de capacidad igualada. El visor usa posiciones originales de
soma: sus líneas son aristas del grafo, no trazados de neuritas. La actividad
mostrada es calculada por el modelo, adimensional y no fisiológica.

Presupuesto común: semillas 0,1,2; 40.960 pasos iniciales y 20.480 pasos por
cada rama sana/lesión de neuronas/lesión de conexiones. Ambas lesiones son del
50% y permanentes. PPO: cinco épocas, evaluación determinista de diez episodios
cada 5.120 pasos. La misma política inicial se copia para todas las ramas y se
reinicia Adam de igual forma en sana y dañadas. La evaluación usa los mismos
mapas, posiciones y orientación por semilla. No hay selección por mejor
checkpoint ni regeneración de neuronas/conexiones.

Ejecutar desde el proyecto:

```sh
python -m flyheal.experiment --config configs/malecns_quick.json --out results/malecns_nuevo
python -m flyheal.report --results results/malecns_nuevo
```

`episodes.csv` conserva retornos crudos y añade `foods_collected`, `collisions`,
`success`. `task_metrics.csv` agrega cada punto por semilla y marca etapa
pre/post/retraining/recovered/healthy. `summary.csv` y `aggregate.csv` mantienen
las métricas de recuperación del retorno; el umbral 90% se deja sin valor cuando B no es positivo. Los resultados requieren leer también las métricas de la tarea.
Con tres semillas y una tarea sintética, las diferencias observadas no
establecen superioridad general ni neuroplasticidad biológica.

## Comparación sintética secundaria

`configs/foraging_quick.json` conserva la comparación original entre densa,
modular dispersa sintética y aleatoria con iguales grados, con ancho 64. Sus
módulos y geometría ilustrativa no representan regiones anatómicas. Se guarda
en `results/foraging_quick`; sus métricas no se atribuyen a MaleCNS ni se mezclan
con la comparación principal. La versión actual contiene únicamente el mundo 2D.

La única opción de `environment` es `foraging`, también utilizada al omitirla. La web y los comandos entrenan exclusivamente la mosca en mundo 2D. Las corridas anteriores fueron eliminadas.
