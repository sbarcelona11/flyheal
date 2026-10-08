# FlyHeal: diseño y plan de implementación

Pedido autorizado: MVP hoy, CPU, mundo 2D, redes densa/estructurada/aleatoria,
lesiones, reentrenamiento, múltiples semillas y documentación en español.

## Diseño
PPO de Stable-Baselines3; extractor compartido actor/crítico: 4→64→64,
tanh y salidas lineales de política/valor. La segunda capa es densa o lleva
una máscara binaria permanente. Estructurada: cuatro módulos de 16 nodos,
conexiones completas intramódulo y cuatro puentes por nodo al módulo siguiente.
Control aleatorio: intercambios de aristas que preservan grados de entrada/salida
y cantidad exacta de conexiones. Es un grafo sintético inspirado en conectomas.
Lesiones uniformes de nodos de la segunda capa (silenciamiento después de tanh)
o de aristas activas de esa capa. Máscaras persistentes en checkpoints; no
regeneración. Copias independientes del agente prelesión para cada intervención.
Control sano continúa con el mismo presupuesto y optimizador reiniciado en
cada rama, incluida la sana. Evaluación determinista con semillas de episodios
reservadas y emparejadas. Tres semillas de entrenamiento; errores estándar
entre semillas, sin presentar episodios como réplicas independientes.

## Plan en esta sesión
1. Escribir y ejecutar pruebas de máscaras, grados, lesión permanente y métricas.
2. Implementar modelos, métricas, runner PPO y serialización.
3. Probar extremo a extremo con entrenamiento corto y recarga de checkpoints.
4. Ejecutar configuración CPU: 3 arquitecturas × 3 semillas, lesiones de 50%
   de nodos o aristas, recuperación y continuación sana de 20.480 pasos.
5. Generar CSV crudo/resumen, gráficos y checkpoints para demo reproducible.
6. Escribir propuesta inicial, informe con resultados reales y guion de 8 minutos.
7. Revisión independiente, correcciones, pruebas finales y empaquetado.

## Criterios de entrega
Sin afirmaciones de superioridad ni novedad global no demostradas; sin conectoma
real. El informe reconoce presupuesto corto, tres semillas, posible efecto techo,
capacidades distintas de la red densa y escasez de un solo patrón de lesión por
semilla. MaleCNS queda como integración futura con procedencia y validación.

## Ajuste exploratorio tras el piloto
La corrida quick mostró políticas finales imperfectas y ablaciones de aristas
beneficiosas. Se conservó completa y se añadió extended.json: 102.400 pasos
iniciales, 40.960 adicionales por rama, checkpoints cada 10.240 y 10 épocas PPO.
Mismas semillas, modelos y lesiones. Este cambio se decide después de observar
el piloto; la segunda corrida sigue siendo exploratoria, no confirmatoria.
No se seleccionan mejores checkpoints ni se ocultan resultados desfavorables.
Se conservan snapshots del código ejecutado antes del endurecimiento posterior
de validación de condiciones duplicadas, sin cambios al cálculo de entrenamiento.
