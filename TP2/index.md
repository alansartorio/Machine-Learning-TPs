---
title: TP2
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

Clasificacion con Arboles de decision. 

---

## Analisis del conjunto de datos

Contamos con datos sobre personas que solicitaron creditos bancarios en basncos alemanes en el año 1994.
Para cada entrada tenemos 20 variables, entre las que se encuentran: 

- creditability: si devolvió el crédito (1) o no (0).
- purpose: Toma valores de 0 a 10, indicando el objeto que el cliente desea comprar, por ejemplo 0 es un auto.
- age..years: edad variable numérica.

Entre otras. 

---

### Que queremos saber?

Queremos poder determinar si una persona devolvera un credito que pide o no.

---

### Que paso en 1994?

Para dar contexto. En 1994 hubo una crisis del marcado de bonos, lo que hizo que aumenten las tasas y la volatilidad del marcado de bonos en todo el mundo.
Todo el mundo vendia los bonos por su gran redito lo cual provoco la mayor crisis financiera en el mundo para inversiones de bonos.

Ahora podemos entender el por que es importante saber si devolveran el credito.

---

### Que variables son cruciales en este contexto historico?

Antes de realizar el analisis fino, podemos tener en cuenta que las siguientes variables pueden ser importantes:

- type.of.apartment: valores de 1 a 3 que son :Free, Rented, Owned.
- no.of.credits.at.this.bank: valores de 1 a 4
- occupation: valores de 1 a 4, Unemployed, Unskilled Permanent Resident, Skilled, Executive.
- no.of.dependents : toma valores 1 y 2, más de 3 propiedades o menos de 3. telephone: toma valores 1 o 2, (sí o no)
- Svalue.savings.stocks: dinero ahorrado, toma valores de 1 a 5, 1 = nada, 2, ≤100, 3, (100,500], 4 (500, 1000].

Por que? Ya que todas dan a entender la situacion financiera del cliente tomando un credito y si este podria ser embargado de no devolver el credito.

---

## Clasificacion

Para la clasificacion primero dividimos el conjunto de datos aleatoriamente en:

- training: 70%
- test: 30%

---

### Arbol de decision

Es un algoritmo de clasificacion que consiste en la creacion de un arbol que categoriza los atributos de los datos, y cada nivel discrimina por el atributo que mas separacion de datos tiene.

---

### Algoritmo

Para poder realizar una clasificacion con el arbol de decision, primero desarrollamos el algoritmo ID3, para su construccion.

---

#### ID3

Es un algoritmo recursivo que va construyendo un arbol segun los atributos de los datos. Como sabe que atributo tiene que usar para discriminar los datos? Se calcula una funcion de ganancia para saber que tanto separa las clases cada atributo y se toma el que mas las separa.
Para este algoritmo de ganancia, se uso la Entropia de Shannon.

---

##### Entropia de Shannon

Se usa para medir la incertidumbre de una fuente de informacion.

$$
H(X) = -\sum_{i=1}^{n} P(x_i) log_b P(x_i)
$$

Donde:

- $n$ es la cantidad de valorees distintos de la variable
- $P(X_i)$ es la frecuencia relativa del $x_i$

---

Esta funcion de entropia luego se usa en la formula de ganancia:

$$
Gain(S,A) = H(S) - \sum_{v \in Valores(A)} \frac{\|{S_v}\|}{\|{S}\|} H(S_v)
$$

---

#### Volviendo a ID3

Una vez implementado, obtuvimos este arbol de decision:

![](./plots/graphs/single_tree_depth_2.svg)

---

#### Random Forest

Ademas de probarlo con un solo arbol, tambien observamos que resultado obteniamos al utilizar 16 arboles y elegir la categoria mas frecuente.

---

## Analisis del modelo

Finalmente para saber que tan bueno es el modelo entrenado, se realizó:

- La matriz de confusion para un arbol y para el random forest.
- Grafico de curvas de precision del arbol teniendo en cuenta la cantidad de nodos, donde en nuestro caso, usamos la altura del arbol.


---

### Presición variando Cantidad de arboles

![](./plots/part1_precision_over_tree count.svg)

---

### Presición variando Tamaño de bolsa

![](./plots/part1_precision_over_bag size.svg)

---

### Matriz de confusion para Arbol simple

![](./plots/part1_single_confusion.svg)

---

### Matriz de confusión para Forest

![](./plots/part1_forest_confusion.svg)

---

### Curvas de precision

![](./plots/part1_single_vs_forest_precision_over_max depth.svg)



# Ejercicio 2

Clasificacion con KNN

---

## Analisis del conjunto de datos

En este caso, como conjunto de datos contamos con 257 registros de opiniones de usuarios sobre una aplicacion, con variables como:

- Review Title
- Title sentiment
- SentimentValue

Entre otros.

---

### Que queremos saber?

Buscamos clasificar opiniones en positivas o negativas segun el sentimiento expresado en el texto del titulo y la descripcion.

---

## Clasificacion

Para clasificar los datos utilizaremos el algoritmo de KNN y KNN con distancias pesadas.

Pero, como funciona KNN

---

## KNN

Es un algoritmo de clasificacion donde para determinar las clases, se ubican los datos en un espacio de N dimensiones y se calculan las distancias entre estos puntos.

Pero como se obtiene la clase resultado? Se seleccionan los $k$ vecinos mas cercanos y se analiza que clase es la mas abundante en ese conjunto.

Esto es el llamado Aprendizaje basado en instancias, ya que para cada dato nuevo necesitamos recalcular su relación con todo el dataset si queremos clasificarlo.

---

## Implementación

Para implementar el algoritmo KNN ubicaremos las variables en el espacio $\mathbb{R}^n$, donde $n$ es la cantidad de variables presentes en el dataset ($n=4$ en nuestro caso). 

Luego, para calcular la distancia entre dos puntos tomaremos la distancia Euclídea.


$$
d(P,Q) = \sqrt{(p_1-q_1)^2 + (p_2-q_2)^2 + \dots + (p_n-q_n)^2}
$$

---

### Clasificación

Luego, teniendo como conjunto de entrenamiento a una lista de valores con formato
$$
(X, f(X))
$$
donde $X=(x_1,x_2,\dots, x_n)$ es un dato del conjunto y $f(X)$ su clase correspondiente. 
Para clasificar una nueva instancia $P$, buscamos los $X_1, X_2,\dots, X_k$ más cercanos a $P$ del conjunto de entrenamiento y la clase que se le asignará a $P$ será la más frecuente dentro de este conjunto, es decir que estimamos la clasificación $\hat f(P)$ como:

$$
\hat f(P) = \underset{c\ \in\ C}{\text{argmax}}\sum_{i=1}^k 1_{\{\ c = f(X_i)\ \}}
$$

---

## Análisis de datos

---

### Cantidad de datos por valoración

![](./plots/part_2/count_per_star_rating.svg)

---

### Promedio de palabras por valoración

![](./plots/part_2/wordcount_per_star_rating.svg){.r-stretch}

Para el caso de las valoraciones de 1 estrella **la cantidad promedio de palabras es de 12.47**

---

### Distribución de datos en el espacio

Para entender un poco mejor la distribución de los datos en el espacio multivariado decidimos reducir la dimensión del dataset a 2 variables para así poder graficar la distribución. 

De esta manera, graficamos los puntos tomando todos los pares de variables posibles. De estos, encontramos que los mejores resultados se dan al relacionar la variable de la cantidad de palabras de una reseña.

---

### Distribución de datos en el espacio

Uno de los resultados más interesantes es el de la relación entre la cantidad de palabras y el sentimiento del texto de la reseña

:::{.container .r-stretch}
::::{.flex-2}
![](./plots/part_2/points_wordcount_textSentiment_swarm.svg)
::::
::::{.flex-1}
En este caso se utilizó un swarmplot para poder apreciar mejor la distribución de los puntos en cuanto a su 'textSentiment'
::::
:::


---

### Distribución de datos en el espacio

En el gráfico se puede apreciar como, dentro de los comentarios de clasificados con sentimiento positivo, los que terminan teniendo mejor puntaje son los de menor longitud.

---

### Distribución de datos en el espacio

El segundo resultado más relevante es la relación entre la cantidad de palabras y la valuación del sentimiento.

![](./plots/part_2/points_wordcount_sentimentValue_scatter.svg)

---

## Elección de k

Para elegir el mejor valor de $k$, corrimos el algoritmo de KNN para todas las instancias del dataset de test probando con todos los valores de $k$ posibles. En base a esto calculamos el porcentaje de aciertos para cada valor de $k$ elegido.

:::{.container .r-stretch}
::::{.flex-2}
![](./plots/knn.svg)
::::
::::{.flex-1}
En base a los resultados obtenidos, el mayor grado de efectividad se obtiene con $k=9$, con el cuál obtuvimos una precisión de 82.35%
::::
:::

<<<<<<< HEAD
En base a los resultados obtenidos, el mayor grado de efectividad se obtiene con $k=9$, con el cuál obtuvimos una precisión de 82.35%

---

## Resultados

<!-- Get plots from part2_result_analysis.py -->

### Matriz de Confusión

![](./plots/part_2/confusion_matrix.svg)

---

## Resultados

### Métricas

![](./plots/part_2/metrics.svg)
=======
# GRACIAS
>>>>>>> 8543fec5cbdb0d2a18851e12e3d2972993e10ed9
