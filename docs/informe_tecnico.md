> Estado actual: versión exclusiva de mundo 2D, sin entrenamientos guardados. Las comprobaciones históricas descritas en este documento deben revalidarse para una nueva entrega.

# Informe técnico — FlyHeal: navegación 2D y recuperación ante daño neuronal

## Resumen

Implementamos un agente PPO que busca tres alimentos y evita obstáculos en un
mundo 2D con ocho señales sensoriales. Comparamos una red densa, un circuito de conectividad medida
MaleCNS y un control recableado de iguales grados. Aplicamos daños
permanentes del 50% de unidades o conexiones, evaluamos antes e inmediatamente
después y reentrenamos frente a una continuación sana. Tres semillas por
arquitectura; los resultados y checkpoints se generarán con nuevas corridas. La navegación es
parcial: recoge alimento en promedio, pero el éxito completo es bajo y variable.
No concluimos superioridad de una arquitectura.

**Procedencia:** se utiliza un subgrafo real de MaleCNS v1.0: 80 neuronas,
1.296 conexiones dirigidas y 26.029 contactos. Todas las conexiones internas,
identificadores, tipos, coordenadas somáticas y predicciones de transmisor se
verificaron contra las tablas oficiales completas: 16 comprobaciones superadas.
La selección procede de una versión fijada de Fly Dino; se validó de forma
independiente sin ejecutar su constructor. SHA-256 y atribuciones CC BY 4.0
acompañan la entrega. Ver [procedencia](procedencia_malecns.md).

La actividad visualizada es calculada por nuestro modelo, no medida en una
mosca. El subgrafo visual→descendente no constituye un circuito olfativo o de
forrajeo validado ni una simulación del cerebro completo.

## Problema y objetivos

Estudiar robustez y recuperación del control con daños internos permanentes,
mediante una aplicación de búsqueda de alimento con observación parcial.
Medir retorno, alimento recolectado, colisiones y éxito; comparar contra una
continuación sana y un control disperso aleatorizado; demostrar el protocolo
mediante una interfaz que sincroniza conducta y activaciones.

## Fundamentos y relación con IA II

PPO actor-crítico optimiza una política estocástica usando retorno descontado,
ventaja y una función de valor. La evaluación utiliza acciones deterministas.
El estado completo del simulador incluye mapa, posición y alimentos; la política
recibe una observación parcial y es feedforward. No incorporamos memoria
recurrente para resolver la observabilidad parcial. La estructura de conexiones
impone restricciones a la red, pero no permite inferir función biológica.

## Entorno y datos

Arena 10×10, cinco obstáculos rectangulares, tres alimentos y límite de 500
pasos. Cada reset genera un mapa reproducible por semilla. Acciones discretas:
girar a izquierda, avanzar o girar a derecha; todas mueven 0,18 unidades y los
giros cambian 0,35 radianes. Observación: olor y distancia libre en tres
direcciones, seno y coseno de orientación. El agente no recibe el mapa completo.

Recompensa: 10 por alimento, 2 veces el avance hacia el alimento más cercano,
−0,01 por paso y −0,15 por colisión. El progreso usa información global del
simulador para calcular recompensa, aunque la política tiene sensores locales.
La colisión se comprueba en la posición final de cada paso, sin barrido continuo:
puede omitir roces de esquinas entre posiciones. No es física de vuelo validada.
El retorno tiene signo y no tiene techo 500.

Las transiciones de tarea se generan mediante interacción. La conectividad y
las anotaciones neuronales provienen del dataset externo MaleCNS v1.0. Evaluación
independiente: semillas 100000 + 100×semilla de entrenamiento + episodio,
emparejadas entre arquitecturas y ramas. Mismos mapas no significan nuevas
réplicas independientes de entrenamiento. No se seleccionó el mejor checkpoint
por evaluación: se usó el final del presupuesto.

## Arquitecturas y metodología

**Circuito MaleCNS:** una población de 80 células, cinco iteraciones internas
por decisión, fuga 0,5 y tanh. Se reinicia el estado a cero en cada observación;
no conserva memoria entre decisiones. Los contactos medidos se normalizan por
el total anatómico entrante; el signo presináptico se supone a partir de la
predicción de transmisor (acetilcolina +1, GABA/glutamato −1, desconocido 0).
Se aprende una ganancia positiva por arista, exp(log_gain) acotada a exp(±3),
sesgos y ganancias de entrada. 24 aristas permanecen anatómicas pero tienen
peso modelado cero: 1.272 con contribución sináptica no nula.

Las ocho observaciones se asignan artificialmente a ocho canales de tipos
visuales (cuatro células por canal); no se afirma que estas células procesen
olor biológico. El actor y crítico PPO son lectores entrenados sobre las 80
actividades, no solo las 16 células anotadas como salidas descendentes.

**Control recableado:** mismos 80 nodos, signos, dinámica y parametrización;
intercambios de destinos preservan grados de entrada/salida y el multiconjunto
de contactos salientes de cada fuente. No se afirma muestreo uniforme. El total
entrante de contactos por destino puede cambiar; la normalización se recalcula.
Las aristas del control son artificiales aunque sus nodos y somas sean reales.

**Densa:** extractor 8→80→80, tanh, lectores lineales. 7.524 parámetros
almacenados contra 1.708 en cada circuito. Tiene dos poblaciones y distinta
dinámica; es una referencia de capacidad, no una comparación equiparada. La
comparación más directa de topología es MaleCNS frente a recableado. Se usan
matrices densas enmascaradas; no se afirma ahorro por kernels dispersos.

PPO en CPU, un hilo, cinco épocas, rollout 512, batch 64, tasa 0,0003, gamma
0,99 y GAE 0,95. Semillas 0,1,2, 40.960 pasos iniciales, 20.480 por cada rama
sana/lesión de neuronas/lesión de conexiones; diez episodios de evaluación cada
5.120 pasos. Cada rama parte del mismo checkpoint y reinicia Adam de forma
simétrica. El control sano recibe igual presupuesto.

El daño se selecciona uniformemente y permanece: se silencia el 50% de células
o se anula el 50% de aristas anatómicas. En los circuitos, una célula silenciada
queda a cero en cada iteración, con conexiones entrantes y salientes anuladas.
En la densa el daño afecta la segunda capa; esto también limita comparabilidad.
No se lesiona el lector PPO. La pérdida de aristas puede incluir las que ya
poseían signo cero; se registran tanto conexiones anatómicas como modeladas.

## Métricas

Retorno B antes, D tras daño, R recuperado, H control sano. Caída=B−D,
ganancia=R−D, brecha sana=H−R. Fracción recuperada=(R−D)/(B−D), solo si B>D;
puede ser negativa o mayor a uno. Denominadores pequeños producen valores
extremos y deben interpretarse con cuidado. El primer checkpoint ≥90% de B
solo se define si B>0, y no exige recuperación sostenida.

Además: media de alimento (0–3), colisiones por episodio y éxito, definido como
recoger los tres alimentos. Las medias se calculan primero por semilla y luego
entre semillas. Barras/bandas ±1 error estándar, no intervalo de confianza del
95%. Ver valores individuales y conteos válidos, especialmente en fracciones.

## Resultados pendientes tras limpiar entrenamientos

Se eliminaron las corridas anteriores a pedido del equipo. Esta versión contiene el código y los datos anatómicos validados, pero no checkpoints ni métricas de una nueva corrida. Antes de completar la entrega final se deberá entrenar, comprobar que la política sana recolecta alimento y ejecutar la comparación con lesiones. No se atribuyen resultados anteriores a esta versión.

Los gráficos e informes se generan al finalizar un nuevo entrenamiento; las cifras deben provenir de sus CSV y metadatos. Las pruebas usan modelos temporales y no constituyen resultados del trabajo.

## Interfaz y validación

MaleCNS se visualiza como una población en sus coordenadas somáticas originales,
centradas y escaladas para la pantalla. Los trazos unen somas de células conectadas;
no son axones o dendritas reconstruidos. Las tarjetas distinguen categorías seleccionadas de entrada visual,
interneuronas y salidas; los puntos usan verde/naranja por signo de actividad.
El contexto gris muestra 124.209 somas reales sin simular, no regiones completas.
El inspector muestra ID, tipo, predicción de neurotransmisor y actividad simulada
adimensional. En las arquitecturas sintéticas se conservan módulos A–D ilustrativos.

Brillo=|actividad|; cruces=lesión permanente. Actividad baja <0,1 en un paso no
implica una lesión. Las líneas discontinuas indican conexiones originales
perdidas. Girar cambia la vista, no el circuito. Cerebro y conducta provienen
de la misma inferencia del checkpoint, separados de los promedios de evaluación.

El protocolo se prueba con entorno, determinismo, sensores, carga y hash de
grafo, grados del control recableado, silenciamiento en todas las iteraciones,
serialización PPO, métricas, API, worker y replay. Ver evidencia final y alcance
en [verificacion_cerebro_2d.md](verificacion_cerebro_2d.md).

## Limitaciones, conclusiones y futuro

Tres semillas y un patrón de lesión por condición, presupuestos modestos,
aprendizaje inestable, observación parcial sin memoria y navegación simplificada.
Hay factores de confusión de capacidad, dinámica y alcance del daño al comparar
con la densa. El subgrafo fue seleccionado para otra tarea visual. Signos de
transmisor, dinámica de tasa y acoplamiento al mundo 2D son supuestos no validados
fisiológicamente. El lector completo permite aprovechar actividad de todas las
células, no exclusivamente sus salidas anotadas. La continuación sana
puede empeorar; la fracción recuperada puede exagerarse con caídas pequeñas.

El aporte es un protocolo reproducible con daño persistente, controles y
visualización de activación real de redes artificiales. No se demuestra
neuroplasticidad biológica, reparación de tejido, función de áreas cerebrales
ni ventaja general de conectividad inspirada en conectomas.

Futuro: ampliar las semillas y patrones de lesión, seleccionar un circuito de
forrajeo con criterios anatómicos y funcionales, comparar lectores restringidos
a neuronas descendentes, calibrar dinámica y signos con evidencia fisiológica,
y agregar morfologías reales y memoria temporal. Las conclusiones presentes se
limitan al modelo y al presupuesto ejecutado.

## Referencias y antecedentes

- [awesome-fly](https://github.com/cobanov/awesome-fly): colección de inspiración.
- [PPO](https://arxiv.org/abs/1707.06347): fundamentos.
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html): implementación.
- [Gymnasium](https://gymnasium.farama.org/): API de entornos.
- [MaleCNS](https://male-cns.janelia.org/download/): datos anatómicos usados, con validación y atribución adjuntas.

