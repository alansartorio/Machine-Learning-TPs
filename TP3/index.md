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

Algoritmo de aprendizaje supervisado que se utiliza para clasificar un conjunto de datos en dos categorías, utilizando la función de activación de la función escalón unitario.

---

### Implementacion

Se utilizó la siguiente función escalón:

![](./plots/step_function.svg){.r-stretch}

<!------->

<!--### Entrenamiento-->

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

![](./plots/tp3-1.svg){.r-stretch}

---

### Resultados perceptron simple

:::{.container .r-stretch}
::::{.flex-1}
![](./plots/ej1.a.gif)
::::
::::{.flex-1}
![](./plots/ej1.a.error.svg)
::::
:::

---

### Es un hiperplano optimo?

Como podemos ver, si bien los datos se separan correctamente, ya que son linealmente separables, el hiperplano no es optimo, ya que no es el que maximiza la distancia entre las dos clases.

<!-- TODO: Mostrar foro que muestre que no es optimo, marcar la distancia entre las clases y mostrar que no es la maxima. -->

---

### Hiperplano Óptimo

Al aplicar el postprocesado al resultado del perceptron simple, se consigue un mejor margen:

![](./plots/post_processed.svg){.r-stretch}

---

## Dataset TP3-2

![](./plots/tp3-2.svg){.r-stretch}

---

### Resultados perceptron simple

:::{.container .r-stretch}
::::{.flex-1}
![](./plots/ej1.c.gif)
::::
::::{.flex-1}
![](./plots/ej1.c.error.svg)
::::
:::

---

### Resultados SVM

<!-- TODO: Agregar foto del perceptron --->

---

### En que se diferencian?

<!-- TODO: Esto-->

---

# Ejercicio 2

Dadas las imagenes provistas, entrenar un SVM para clasificacion de los pixeles en 3 clase utilizando limites de decision no lineales.

---

## Por que no funciona el SVM?

En este caso, no tenemos una separacion lineal de las clases. No existe hiperplano capaz de separar las clases correctamente.

---

## Que podemos hacer?

Para solucionar este problema, podemos utilizar un kernel que nos permita mapear los datos a un espacio de mayor dimension, donde si exista una separacion lineal.

---

## Kernel

Un kernel es una funcion que nos permite mapear los datos a un espacio de mayor dimension, donde si exista una separacion lineal.

Se aplica a la funcion de decision de la siguiente manera:

$$
f(x) = \sum_{i=1}^{n} \alpha_i K(x_i, x) + b_0
$$

---

### Tipos de kernel

- Lineal
  $$ K(x, x') = x \cdot x' $$
- Polinomial
  $$ K(x*i, x') = (1+\sum^{p}*{j+1}x\_{ij} \cdot x_j)^d $$
- Radial
  $$ K(x*i, x') = e^{-\gamma \sum^{p}*{j+1}||x\_{ij} - x_j||^2} $$

---

### Como haremos la comparacion?

Como sabemos los SVMs solo pueden compara 1 vs 1 y en este caso tenemos 3 clases.
Para solucionarlo utilizamos ....

<!-- TODO: AGREGAR COMO HACEMOS LA COMPARACION -->

---

## Datos

<!-- TODO: Imagenes de los datos -->

---

### Conversion de las imagenes a vectores

<!-- TODO: Explicar -->

---

### Division en training y test

<!-- TODO: Explicar -->

---

## Resultados

---

### Resultados con kernel lineal

<!-- TODO: Matriz de confusion -->

---

### Resultados con kernel polinomial

<!-- TODO: Matriz de confusion -->

---

### Resultados con kernel radial

<!-- TODO: Matriz de confusion -->

---

### Cual da mejores resultados?

<!-- TODO:  -->

---

## Analisis de resultados

Veremos como funicona el SVM con las imagenes dadas.

---

### Imagen cow.jpg

<!-- TODO: Agregar imagen -->

---

### imagen vaca.jpg

<!-- TODO: Agregar imagen -->

---

### NO SE A QUE SE REFIERE EL h)

---

---

# GRACIAS
