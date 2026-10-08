# Datos y referencias externas

El código propio de FlyHeal se distribuye bajo MIT (LICENSE). Los datos MaleCNS
incluidos se distribuyen bajo CC BY 4.0, con atribución a FlyEM/HHMI Janelia,
University of Cambridge, MRC Laboratory of Molecular Biology y Google Research.
Ver `flyheal/data/malecns80/NOTICE.md`, el manifiesto y la validación. El atlas
se generó independientemente de las anotaciones oficiales; su selección se
inspira en el atlas de Fly Dino y no representa una segmentación completa.

La selección previa de 80 células procede de `cobanov/flyjump`, fijada al commit
`e34c6614e7d13a1018585f705cdeb59b9f38291d`, por Mert Cobanov. Se conservaron los
bytes originales del grafo y se verificaron contra todas las tablas oficiales.

El diseño de actividad cerebral y red de decisión se inspira en Fly Dino:
https://github.com/cobanov/flyjump . Los componentes del visor FlyHeal fueron
implementados para este proyecto; no se trasladó código de sus componentes ni
se incorporó su modelo corporal. Se mantiene referencia visible al proyecto y
autor. FlyHeal muestra el lector PPO real 80→3 lineal; no copia su lector 16→12→3,
su entrenamiento CEM ni sus resultados.

Gymnasium, Stable-Baselines3, PyTorch, NumPy, pandas y Matplotlib conservan sus
licencias respectivas; se instalan como dependencias, no se redistribuyen aquí.
