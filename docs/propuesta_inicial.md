# Propuesta inicial — Inteligencia Artificial II

**Título:** FlyHeal: robustez y recuperación de una mosca virtual con aprendizaje por refuerzo y un subgrafo real de MaleCNS.

**Integrantes:** Mathias Rodriguez Agro y Sebastian Barcelona.

## Tema y aplicación

Un agente virtual busca alimento y evita obstáculos en un mundo 2D. Investigamos qué ocurre con su desempeño cuando pierde permanentemente unidades o conexiones de su red, y cuánto recupera al continuar aprendiendo. La aplicación es navegación y control tolerante a fallas con sensores limitados.

La comparación principal utiliza un subgrafo real de MaleCNS v1.0: 80 neuronas, 1.296 conexiones dirigidas y 26.029 contactos sinápticos. Sus identificadores, tipos, posiciones de soma, predicciones de neurotransmisor y contactos fueron verificados contra las tablas oficiales completas. Se compara con un control recableado que conserva grados dirigidos y los contactos asociados a cada origen, y con una red densa de ancho 80 como referencia de capacidad. Esta última tiene dos capas y no iguala capacidad ni dinámica con el circuito.

El subconjunto se seleccionó previamente de vías visuales hacia neuronas descendentes para un experimento de Dino. Reutilizarlo para forrajeo no lo convierte en un circuito olfatorio. Los ocho sensores, su inyección en 32 células y la lectura aprendida de las 80 actividades son decisiones de ingeniería. La comparación modular sintética está disponible como configuración secundaria del mismo mundo 2D. La entrega parte sin entrenamientos guardados.

## Relación con IA II

Aprendizaje por refuerzo, política, valor, retorno descontado y PPO actor-crítico. La observación sensorial no incluye el mapa completo: complementamos la asignatura con observabilidad parcial, restricciones de conectividad, ablaciones permanentes, controles experimentales y reproducibilidad. El circuito real calcula cinco iteraciones internas por decisión y reinicia su estado en cada observación; no conserva memoria entre pasos del entorno.

## Objetivos

**General:** implementar y evaluar un agente RL de navegación con daños internos permanentes y reentrenamiento, usando conectividad medida como restricción estructural.

**Específicos:**

- Entrenar MaleCNS, su control recableado y la referencia densa con presupuesto e hiperparámetros comunes.
- Lesionar el 50% de unidades o conexiones elegibles y mantener la lesión durante el reentrenamiento.
- Medir retorno, alimento recolectado, colisiones y éxito antes, después y durante recuperación.
- Comparar con una continuación sana del mismo checkpoint y con el control recableado.
- Ejecutar tres semillas y mostrar actividad calculada del modelo en una demo interactiva.
- Mantener procedencia, licencia, comprobaciones y límites científicos visibles.

## Datos o entorno

Entorno propio de Gymnasium: arena 10×10, obstáculos rectangulares, tres alimentos y horizonte de 500 pasos. Las acciones son girar a izquierda, avanzar o girar a derecha; todas producen movimiento. Ocho observaciones: señales locales de olor y distancia a obstáculos a izquierda/frente/derecha, más seno y coseno de orientación. El agente no recibe posiciones absolutas de alimento ni mapa. La recompensa incluye alimento, costo temporal, colisiones y progreso; este último utiliza distancia global calculada por el entorno, una simplificación explícita.

Las experiencias se generan mediante interacción; la conectividad procede de datos externos MaleCNS bajo CC BY 4.0. Guardamos métricas por episodio, semillas, mapas reproducibles, retornos, alimento, colisiones, éxito, máscaras, configuración y checkpoints. La evaluación usa semillas reservadas y mapas emparejados entre ramas y arquitecturas. La atribución y verificación están en `docs/procedencia_malecns.md` y `flyheal/data/malecns80/`.

## Metodología

Python, Gymnasium, PyTorch, Stable-Baselines3 PPO, NumPy, pandas y Matplotlib en CPU. MaleCNS usa contactos medidos normalizados, signos simplificados según neurotransmisor, ganancias entrenables y cinco iteraciones de tasa tanh con fuga. PPO aprende la lectura actor-crítico de las 80 células. Los 16 índices descendentes originales se conservan como metadatos, sin restringir esa lectura a ellos. El control recableado conserva número de aristas, grados de entrada/salida, multiconjunto de contactos por origen, etiquetas, inyección y dinámica; no conserva contactos entrantes por destino ni la topología biológica. La referencia densa 8→80→80 tiene otra capacidad y dinámica.

Lesiones permanentes, ramas que parten del mismo checkpoint y reinicio simétrico de Adam. `configs/malecns_quick.json` fija 40.960 pasos iniciales, 20.480 por rama y 10 episodios por checkpoint; tres semillas por arquitectura. Se publican media y dispersión entre semillas, con resultados de la tarea además del retorno. La comparación sintética 8→64→64 permanece en `configs/foraging_quick.json` como referencia separada.

## Viabilidad y entregables

MVP CPU con código modular, configuraciones, datos por episodio, gráficos, checkpoints, README, propuesta, informe y guion de ocho minutos. La demo sincroniza mundo y actividad del modelo. Las posiciones reales son somas; las líneas entre ellas son conexiones de grafo, no reconstrucciones de neuritas. El brillo muestra actividad simulada adimensional y no una medición fisiológica.

Tres semillas y un patrón de lesión por condición sustentan una exploración inicial. Los datos reales no validan por sí solos la conducta, los signos, la dinámica o una reparación biológica. La comparación principal permite estudiar si la topología seleccionada aporta algo frente a recablearla; la densa aporta una referencia adicional, sin demostrar equivalencia de capacidad. Futuro: más semillas, varios patrones de daño, controles adicionales y un circuito sensorial apropiado para la tarea.
