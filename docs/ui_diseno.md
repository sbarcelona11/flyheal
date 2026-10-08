# Diseño de FlyHeal Lab

La corrida principal muestra 80 neuronas en coordenadas somáticas MaleCNS
validadas, en una población. Las líneas representan conectividad entre somas,
no morfología de neuritas. Brillo = magnitud de actividad simulada; colores =
categorías seleccionadas de entrada visual, interneuronas y salidas descendentes.
El inspector muestra ID original, tipo, transmisor predicho, coordenadas y actividad.
No se presenta el subgrafo como cerebro completo ni como circuito de olor validado.

El control recableado conserva coordenadas y tipos, con conectividad artificial;
se rotula explícitamente. Las corridas sintéticas conservan posiciones ilustrativas
y cuatro módulos A–D, entre dos poblaciones artificiales. La etiqueta de procedencia
cambia con la arquitectura, incluso al elegir la densa dentro del estudio MaleCNS.

Cruces = silenciamiento permanente. Líneas discontinuas = aristas perdidas frente
al checkpoint original. En el circuito se anulan entradas y salidas de la célula
silenciada. Baja actividad <0,1 es instantánea y distinta del daño. Se muestran
todas las células y hasta 650 aristas; el contador usa todas las aristas retenidas,
y se informa cuántas tienen peso modelado no nulo.

Mundo y cerebro se reproducen sincronizados desde una inferencia guardada. El
observador ve el mapa completo; la política solo ocho señales. Los promedios
provienen de los episodios reservados, no del episodio individual mostrado.

Entrenamiento desde la web en un proceso CPU propio, con progreso y cancelación.
MaleCNS fija el grafo en 80 células y el entorno 2D; las redes sintéticas admiten
64/80/128/256 unidades por capa y densidad configurable. Cada ejecución conserva
sus archivos y máscaras. Vista adaptable, controles por teclado, giro del circuito,
selector de célula, etapas, cursor, velocidad y modo presentación.

## Lectura de decisión inspirada en Fly Dino

Atlas gris de 124.209 somas reales excluidos del modelo; no se simulan. Actividad
con signo en las 80 células y controles de enfoque/órbita. La decisión muestra
el lector entrenado real 80→3 lineal: suma de actividades×pesos más sesgo, logits,
softmax y argmax determinista. Las líneas muestran hasta 12 aportes por acción;
el cálculo usa todos. Se listan seis aportes mayores a la acción elegida. La vista
es una implementación propia, con referencia visible a Fly Dino/Mert Cobanov.
