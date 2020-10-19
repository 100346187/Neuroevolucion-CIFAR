# Neuroevolucion-CIFAR
El objetivo principal de este trabajo es diseñar, implementar y aplicar un algoritmo genético capaz de construir automáticamente redes neuronales convolucionales que, una vez entrenadas, puedan resolver correctamente una tarea de visión artificial.

## Introducción

El diseño de arquitecturas de redes de neuronas convolucionales para resolver una tarea dada es complejo y requiere un proceso de experimentación casi aleatorio con largos entrenamientos a cada prueba. Esta complejidad en el diseño de arquitecturas neuronales surje por el enorme número de parámetros implicados en su construcción y por la falta de patrones de diseño que ayuden a convertir el problema en una búsqueda dirigida con garantías.

La solución clásica al problema antes mencionado es el uso de capital humano experto en la materia que dedique su tiempo y esfuerzo a dicho proceso de prueba y error intentando mejorar la búsqueda mediante su pericia. Esto hace que la construcción de redes de neuronas convolucionales de calidad sea un proceso lento, tedioso y caro para las empresas y equipos de investigación interesados.

Debido a que la solución clásica no parece resolver completamente el problema del coste del diseño, se han empleado métodos meta-heurísticos para encontrar procesos de búsqueda más eficientes. De entre estos métodos meta-heurísticos destacan los algoritmos genéticos, que al aplicarse al problema de la construcción de redes de neuronas conforman el campo de la neuroevolución. Este es el campo en el que se encuadrará este proyecto.

## Proyecto
Este trabajo está compuesto de tres ficheros de código en Python y un documento en formato PDF:
-ga.py: En este fichero se establecen todas las clases que más tarde se usarán para contruir el algoritmo genético, concretamente la clase Individuo y la clase AlgoritmoGenético.
-main_ga.py: Fichero obtenido a través de la conversión de un fichero propio de cuadernos Jupyter. En este código se construye el algoritmo genético y se realiza su evolución empleando paralelización.
-tf_cifar_val.py: Este fichero se encarga de construir las redes neuronales de forma automática en función de los parámetros que le envía main_ga.py y las evalúa con los datos de validación.
-Tecnicas_de_computacion_evolutiva_aplicadas_al_diseno_de_redes_de_neuronas_profundas.pdf: Memoria del proyecto al completo. Se recomienda su lectura para entender su alcance y metodología.
