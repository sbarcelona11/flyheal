> Corridas anteriores eliminadas. Antes de exponer, ejecutar una nueva comparación y completar la tabla con sus resultados. No hay cifras finales en esta versión.

# Guion de presentación — 8 minutos

Siete pantallas; los tiempos suman 8:00. Presentar la comparación principal `malecns_quick`. Consultar la tabla definitiva del informe técnico y las figuras de esa corrida: no extrapolar los resultados de `foraging_quick` al circuito real. Completar nombres reales del grupo antes de entregar.

## 1. Pregunta y aplicación — 0:00–0:50

Mostrar título y mapa 2D.

«Estudiamos cómo un agente de aprendizaje por refuerzo conserva y recupera su comportamiento después de perder parte de su red. Una mosca virtual busca tres alimentos y evita obstáculos. Usamos una restricción de conectividad tomada de un subgrafo real de MaleCNS: 80 neuronas, 1.296 conexiones dirigidas y 26.029 contactos. Los datos originales fueron verificados. La conducta y el aprendizaje siguen siendo una simulación diseñada por nosotros.»

## 2. Relación con IA II y entorno — 0:50–1:45

Mostrar sensores → PPO → acción → entorno → recompensa.

«Usamos PPO actor-crítico. El agente recibe ocho señales: olor y cercanía de obstáculos en tres direcciones, más orientación. Puede girar o avanzar. La observación es parcial. El circuito calcula cinco iteraciones internas por decisión, pero reinicia el estado para cada observación, sin memoria entre pasos. La recompensa combina alimento, progreso y penalizaciones. El progreso usa distancia global calculada por el entorno, una ayuda de entrenamiento que debemos reconocer.»

## 3. Arquitecturas y procedencia — 1:45–2:50

Mostrar MaleCNS / recableada / densa, con leyenda de procedencia.

«La comparación principal enfrenta el subgrafo MaleCNS con su versión recableada. El control conserva grados dirigidos, número de conexiones y contactos por neurona de origen; cambia qué destinos reciben esas conexiones. Mantiene la misma inyección sensorial y dinámica. La densa tiene dos capas de 80 unidades: sirve como referencia adicional y no iguala capacidad o dinámica.

La selección original es visual hacia neuronas descendentes, no olfatoria. Inyectamos los ocho sensores en 32 células y PPO aprende una lectura de las 80 actividades. Los 16 índices de salida originales son metadatos. Las posiciones reales son somas, no formas de axones o dendritas. Conservamos la comparación modular sintética como experimento secundario.»

## 4. Protocolo de lesión y recuperación — 2:50–4:00

Mostrar antes → ramas sana/unidades/conexiones → evaluación.

«La configuración principal define tres semillas por arquitectura y 40.960 pasos iniciales. Copiamos el checkpoint para una continuación sana y dos lesiones: mitad de unidades elegibles o mitad de conexiones elegibles. La lesión permanece. Todas las ramas reinician Adam de igual forma y reciben 20.480 pasos adicionales. Evaluamos diez episodios por punto con mapas emparejados y semillas reservadas. Las bandas representan variación entre semillas, no diez entrenamientos independientes por evaluación. Usamos el checkpoint final, sin seleccionar el mejor resultado.»

## 5. Resultados de la comparación principal — 4:00–5:30

Mostrar la tabla y figuras definitivas de `malecns_quick` incluidas en el informe técnico. Preparar antes de presentar los valores de alimento, colisiones, éxito y retorno para cada arquitectura y etapa. Si una corrida está incompleta, identificarla como pendiente y no atribuirle números del experimento sintético.

«Leemos primero qué conducta aprendió cada agente antes del daño. Después comparamos la caída inmediata y el resultado del reentrenamiento con la continuación sana. Alimento, colisiones y éxito importan además del retorno, porque la recompensa también contiene progreso. El control recableado permite preguntar si esta topología concreta aporta algo bajo nuestras decisiones de modelado. La densa compara otra arquitectura y capacidad.

Una mejora después de ablación o una recuperación superior al valor inicial es posible y debe mostrarse. La fracción recuperada solo aplica si hubo caída; un denominador pequeño puede producir cifras extremas. Tres semillas y un patrón de daño por condición no establecen superioridad general. Los resultados sintéticos anteriores se presentan por separado.»

## 6. Demo del cerebro y mundo — 5:30–6:30

Preparar `python -m flyheal.ui` antes de exponer. Elegir el experimento MaleCNS, una semilla y una lesión con checkpoints disponibles. Alternar Antes, Daño y Reentrenado; inspeccionar una célula y sus metadatos.

«El mapa y el grafo muestran el mismo episodio. El brillo representa actividad calculada del modelo; baja actividad no significa lesión. Una unidad silenciada queda marcada y las conexiones perdidas permanecen desactivadas. El identificador, tipo y soma vienen de MaleCNS; la actividad es simulada. Las líneas dibujadas conectan somas y no reconstruyen neuritas. Girar la vista cambia la presentación, no los datos. Este episodio individual no reemplaza la evaluación agregada.»

## 7. Límites, conclusión y futuro — 6:30–8:00

Mostrar daño persistente / controles / procedencia.

«El resultado es un MVP reproducible que incorpora conectividad medida, entrena, lesiona, reentrena y evalúa. La recuperación ajusta parámetros supervivientes; no regenera neuronas o conexiones. El control sano es necesario: entrenar más también puede mejorar o empeorar sin lesión.

Los límites son tres semillas, un patrón de daño, distinta capacidad de la densa, observación parcial sin memoria y una tarea simplificada. La recompensa de progreso usa información global y la colisión se comprueba al final del paso. Los signos de neurotransmisor, pesos normalizados, dinámica, sensores y lector de acciones son decisiones de ingeniería. Este subconjunto visual no demuestra un circuito de búsqueda de alimento, fisiología real o reparación biológica.

Entregamos datos y licencia CC BY 4.0, verificación de procedencia, código, configuraciones, métricas, gráficos y checkpoints. El siguiente paso es repetir con más semillas, patrones de daño y un circuito sensorial mejor justificado. La visualización ayuda a inspeccionar el modelo, pero no sustituye evidencia experimental.»

## Preparación y respaldo

Ensayar con cronómetro. Abrir demo, figuras y tabla definitiva antes de exponer. Si falla el navegador, usar capturas y figuras incluidas; no entrenar durante la presentación. Integrantes: Mathias Rodriguez Agro y Sebastian Barcelona.

Preguntas clave: ¿conectividad real? Sí, un subconjunto verificado de MaleCNS. ¿Es olfatorio? La selección es visual; la tarea y sus entradas son artificiales. ¿Qué se recupera? Desempeño mediante ajuste de parámetros supervivientes. ¿Quién gana? Responder con la tabla definitiva y su incertidumbre, sin afirmar superioridad general.

