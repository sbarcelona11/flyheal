# Procedencia y verificación del subgrafo MaleCNS

FlyHeal incluye un subgrafo real de MaleCNS v1.0: 80 neuronas, 1.296 conexiones dirigidas y 26.029 contactos sinápticos. Los identificadores, tipos, posiciones de soma, predicciones de neurotransmisor y cantidades de contactos coinciden exactamente con las tablas oficiales. La red completa tiene muchas más neuronas: este archivo es un subconjunto seleccionado.

Los datos originales pertenecen a FlyEM/HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology y Google Research. Se distribuyen bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); la licencia y atribución se conservan en `flyheal/data/malecns80/NOTICE.md`. La [página oficial](https://male-cns.janelia.org/download/) enumera las tres tablas empleadas y su licencia.

La selección previa procede de [cobanov/flyjump](https://github.com/cobanov/flyjump), commit `e34c6614e7d13a1018585f705cdeb59b9f38291d`. Se conserva `public/data/connectome/graph.json` sin modificar sus bytes. Su SHA-256 es `2424c9dd2e44534e600aeda1a9058039b1f22a4bd983284a27adc10b22130719`.

## Qué se verificó

Se descargaron las tres tablas oficiales completas desde el bucket público `flyem-male-cns/v1.0/connectome-data/flat-connectome` y se ejecutó un verificador independiente, sin ejecutar el constructor externo. Los SHA-256 originales coinciden con los tres valores de la derivación fijada. Se cotejaron los 80 identificadores, tipos, posiciones y neurotransmisores, y se recorrieron las 151.856.684 filas de la tabla de conexiones. Se seleccionaron todas las filas cuyos dos extremos pertenecen a las 80 neuronas y se compararon sus pares dirigidos y cantidades de contactos como multiconjunto: igualdad exacta, sin conexiones internas faltantes ni agregadas.

También se verifican unicidad de identificadores, canales de entrada, cuatro neuronas por cada uno de los ocho tipos visuales, 32 entradas, 16 salidas anotadas como neuronas descendentes y coherencia de los signos asumidos. El resultado reproducible se registra en `flyheal/data/malecns80/validation.json`, con `status: passed`. El cargador puede usar este estado y el hash del grafo para impedir que un archivo distinto se presente como verificado.

| Tabla oficial | Tamaño descargado (bytes) | SHA-256 |
| --- | ---: | --- |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14.483.314 | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1.051.241.946 | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43.282.834 | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |

Los enlaces exactos de descarga están en `manifest.json` y `validation.json`. El total descargado fue 1.109.008.094 bytes, aproximadamente 1,03 GiB. Las tablas originales permanecen en el área de trabajo y no se incluyen en la entrega.

## Alcance científico

La selección previa tomó cuatro células de cada tipo visual LC4, LC11, LC9, LC15, LC16, LC17, LC21 y LPLC2 según sus contactos hacia neuronas descendentes; escogió 16 salidas descendentes y 32 células intermedias, conservando todas las conexiones internas medidas. Esta verificación confirma que los datos de esas células y conexiones son reales; no reproduce de forma independiente la clasificación completa que produjo la selección original.

La tarea de forrajeo introduce señales artificiales en esas entradas y convierte la actividad de salida en acciones mediante un lector diseñado para la simulación. No se ha demostrado que el circuito seleccionado codifique olor, hambre o búsqueda de alimento. Los signos acetilcolina `+1`, GABA/glutamato `−1` y desconocidos/moduladores `0`, la normalización de pesos y las dinámicas de tasa son supuestos del modelo. Las posiciones son posiciones originales de soma; no representan trazados de axones o dendritas. La actividad simulada es adimensional y no constituye una medición neural.

## Reproducir la verificación independiente

Desde la raíz del proyecto, usar Python 3.11 o posterior y `pyarrow` (la corrida
registrada usó Python 3.12 y pyarrow 25.0.1). El verificador no ejecuta código
externo ni modifica el grafo o manifiesto. Los archivos completos requieren
aproximadamente 1,11 GB de descarga y almacenamiento; no deben incorporarse a
los resultados o al repositorio.

```sh
python -m pip install pyarrow
mkdir -p work/malecns_raw
curl -fL https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather -o work/malecns_raw/annotations.feather
curl -fL https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather -o work/malecns_raw/edges.feather
curl -fL https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather -o work/malecns_raw/neurotransmitters.feather
python scripts/verify_malecns.py work/malecns_raw --report work/malecns_revalidation.json
```

El programa devuelve código cero únicamente si pasan todas las comprobaciones.
La opción `--data-dir` permite verificar otra copia de los cuatro archivos del
paquete; `--report` es opcional y guarda un reporte nuevo. El código cero confirma
igualdad con las fuentes fijadas, no valida los supuestos fisiológicos o la tarea.

## Atlas gris de contexto

Se generó de la misma tabla oficial de anotaciones: somas `Traced` de las clases
ópticas, centrales y descendentes definidas en `atlas.json`. La selección sigue
el criterio publicado por Fly Dino, aplicado de forma independiente. Excluye
los 80 IDs simulados: **124.209 puntos grises**, sin actividad ni neuritas. La
selección por clases no es una segmentación anatómica completa. Las coordenadas
originales permanecen sin alterar; el visor centra, escala uniformemente y gira.
`atlas_manifest.json` fija SHA-256 y cuenta; la atribución CC BY 4.0 de NOTICE.md
se aplica también a esos datos.

Para regenerar el atlas desde las anotaciones originales verificadas:

```sh
python scripts/build_atlas.py /ruta/a/los/originales
```

Requiere la misma dependencia opcional pyarrow del verificador. Comprueba el
hash oficial antes de copiar coordenadas, excluye IDs modelados y registra hash
final en atlas_manifest.json. El criterio por superclases y estado `Traced` está
incluido en el archivo; no se inventan posiciones.
