# Modelo de clasificación para la evaluación de Toros 
El siguiente repositorio consiste en una implementación de una técnica de aprendizaje máquina sin el uso de un framework para la prediccion de clases dependiendo de los puntajes de los toros para cada una de sus caracteristicas fisicas.

## Modulos actuales
- **toros.csv** : Es la base de datos proveniente del catalogo abierto de SelectSires para sus sementales junto con sus caracteristicas
- **dset.py** : contiene el proceso utilizado para la limpieza y visualizacion de datos
- **doc.pdf**: Es un informe con la descripcion preliminar del modelo, planteamiento del probelma, el conjunto de datos, implementacion del modelo y los resultados del entrenamiento

/
/
/
/
/

El objetivo es dividir cada tecnica de aprendizaje de maquina por modulo como una mejora:
- main.py : Datos principales y manda a llamar a los demas modulos
- dset.py : limpieza y visualizacion de datos
- gd.py : gradiante descendiente, regresion logistica y metricas
- plot.py : graficacion de cada punto de interes, como el costo y la certeza
- flask.py : pequena interfaz de usuario para el hipotetico productor de lacteos
