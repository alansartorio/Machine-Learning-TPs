---
title: TP3
# subtitle: Aprendizaje Automático
author:
  - Benvenuto Agustín - 61448
  - Galarza Agustín - 61481
  - Sartorio Alan - 61379
# date:
# - 13 de Marzo 2024
# theme: "Copenhagen"
---

# Ejercicio 1

Utilizar los metodos de SVM y perceptron simple para clasificacion de un dataset preparado por nosotros.

---

## Perceptron simple

Algortimo de aprendizaje supervisado que se utiliza para clasificar un conjunto de datos en dos categorías, utilizando la función de activación de la función escalón unitario.

---

### Implementacion

<!-- TODO: Agregar informacion sobre la funcion de activacion --->

---

### Entrenamiento

 <!-- TODO: Agregar informacion sobre los parametros, la funcion de error, como hace para aprender --->

## SVM

Es un algoritmo de aprendizaje supervisado que se utiliza para clasificar un conjunto de datos en dos categorías, utilizando un hiperplano que maximiza la distancia entre las dos clases.

Siendo capaz de generar planos N dimensionales de la forma:

$$
\vec{w} \cdot \vec{x} + b = 0
$$

---

### Implementacion

Para el entrenamiento, tomamos los datos de entrada e iteramos por cada punto en el plano.
Se calcula la distancia de cada punto al hiperplano y se actualizan los pesos y el bias en caso de que el punto este mal clasificado.

Si esta mal clasificado se corrije con:

$$
\vec{w}^{nuevo} = \vec{w}^{viejo} - k (\vec{w}^{viejo} - Cx_iy_i)\\
b^{nuevo} = b^{viejo} - k (- Cy_i)
$$

Si esta bien clasificado cambia menos:

$$
\vec{w}^{nuevo} = \vec{w}^{viejo} - k \vec{w}^{viejo}
$$

---

Ademas, decrementamos el valor de k en cada iteracion para que el algoritmo converja, con una funcion exponencial decreciente.

$$
k_{nuevo} = k_{viejo} * e^{-currentIteration / maxIterations}
$$

---

Finalmente, para intentar no caer en minimos locales, se implemento el algoritmo de entrenamiento para que sea estocastico.

Tomando una cantidad de puntos aleatorios en cada iteracion, mas concretamente un 50%.

<!-- TODO: POR QUE ELEGIMOS ESTE NUMERO, HACER ALGO -->

---

## Dataset TP3-1

<!-- FOTO DEL DATASET -->

---

### Resultados perceptron simple

<!-- TODO: Agregar foto del perceptron --->

---

### Resultados SVM

<!-- TODO: Agregar foto del perceptron --->

---

### Es un hiperplano optimo?

Como podemos ver, si bien los datos se separan correctamente, ya que son linealmente separables, el hiperplano no es optimo, ya que no es el que maximiza la distancia entre las dos clases.

<!-- TODO: Mostrar foro que muestre que no es optimo, marcar la distancia entre las clases y mostrar que no es la maxima. -->

---

### En que se diferencian?

<!-- TODO: Esto-->

---

## Dataset TP3-2

<!-- FOTO DEL DATASET -->

---

### Resultados perceptron simple

<!-- TODO: Agregar foto del perceptron --->

---

### Resultados SVM

<!-- TODO: Agregar foto del perceptron --->

---

### En que se diferencian?

<!-- TODO: Esto-->

---

# Ejercicio 2

---

# GRACIAS
