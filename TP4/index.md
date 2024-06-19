---
title: TP4
# subtitle: Aprendizaje Automático
author:
  - Benvenuto Agustín - 61448
  - Galarza Agustín - 61481
  - Sartorio Alan - 61379
# date:
# - 13 de Marzo 2024
# theme: "Copenhagen"
---

# Métodos de aprendizaje no Supervisados

---

## Introducción

Los modelos de aprendizaje no supervisado son modelos de predicción en los cuáles la variable de respuesta no es conocida. 

En su lugar, se buscan analizar las relaciones entre las variables.

---

## Comparación con métodos supervisados

La principal ventaja de los métodos de aprendizaje no supervisados es que para éstos es más fácil (y barato) conseguir datos de entrenamiento, ya que se evitan los gastos de conseguir y validar la clasificación de los mismos.

---

## Clustering

Es uno de los principales métodos de aprendizaje no supervisado, el cuál trataremos en esta presentación.

Consiste en observaciones de acuerdo a algún criterio con el objetivo de encontrar **subgrupos (clusters)** de observaciones. Además, se busca minimizar la distancia entre las observaciones de un mismo subgrupo, a la vez que maximizamos la distancia entre los elementos de subgrupos diferentes.

---

### Clustering - Métodos

Dentro de los algoritmos de clusterización hay muchos métodos distintos, con diferentes criterios de partición. Dentro de estos, nosotros prestaremos atención a dos de ellos.

:::{.container .r-stretch}

::::{.flex-1}

#### Clusters basados en prototipos
Un cluster es un conjunto de objetos en el cual cada objeto está más cerca (o es más similar) al prototipo que define al cluster, que al prototipo que define cualquier otro cluster.

::::
::::{.flex-1}

#### Agrupamiento jerárquico
Se organizan los puntos de datos en una jerarquía de
clústeres basados en su similitud o distancia. En este caso, se definene subgrupos dentro de subgrupos y esta jerarquía se representa en un dendograma.

::::
:::

---

## K-means

K-means es un método de clustering, basado en prototipos.

Busca dividir al conjunto de datos en $K$ grupos, donde cada grupo es representado por el promedio de los puntos que lo componen. A este promedio se lo denomina _**centroide**_.

El parámetro $K$ es una constante que se fija antes de la ejecución del algoritmo.

---

### K-means - Concepto

Dado un conjunto de observaciones $\{x_1,x_2,\dots,x_n\},\ x_i\in\mathbb{R}^p$

Es decir, un dataset con:

- $n$ documentos
- $p$ variables por documento

El algoritmo K-means construye una partición de las observaciones en $K$ conjuntos ($k\le n$) que minimiza la distancia de los elementos dentro de cada grupo $S_i\ /\ S=\{S_1,S_2,\dots,S_k\}$

---

### K-means - Concepto

En otras palabras, buscamos la asignación de $S$ tal que se minimice la fórmula

$$
\sum_{i=1}^k\sum_{x\in S_i}||x-\mu_i||^2
$$

---

### K-means - medida de calidad

En K-means buscaremos minimizar la variabilidad dentro de los clusteres generados. Esto significa que queremos que los puntos del cluster se organicen de forma compacta alrededor del centroide.

<!-- TODO: añadir el algoritmo, o es demasiado? -->
Para determinar la variablidiad, utilizaremos la métrica de _Within-set-sum-of-squared-errors_ (WSSC).

---

### WSSC

Para un dataset con $n$ observaciones y $p$ variables, al cuál dividimos en $K$ clusters tal que el conjunto de clusters se define como $C=\{C_1,C_2,\dots,C_k\}$. La métrica WSSC se define como:

$$
WSSC= \sum_{k=1}^K\sum_{x\in C_k}||x-\mu_k||^2
$$

---

## Dataset

Para este trabajo, nos fue provisto un conjunto de datos con información sobre películas. Este dataset tiene un total de 14 variables, entre las cuáles se encuentran:

- original title
- imdb id
- budget
- revenue
- vote average
- popularity
- budget 

---

### Análisis de datos

Como primer análisis de los datos provistos, encontramos la presencia tanto de campos vacíos como de documentos "repetidos", los cuáles describían a la misma película (identificada por su _imdb id_) y que en algunos casos contenían información diferente para una misma variable.

Para solucionar esto, obtuvimos acceso a la api de **tmdb**, la cuál consiste en una gran base de datos cinematográfica. En ella se puede encontrar una gran cantidad de películas, identificadas por su _imdb id_.

---

### Análisis de datos

De esta forma, pudimos por un lado rellenar los campos faltantes con la información obtenida de esta api (con excepción de los documentos sin un _imdb id_, los cuáles decidimos eliminarlos al ser una pequeña proporción del total), y por el otro resolver los datos en conflicto de los documentos repetidos al tomar a la información de la api como una única fuente de verdad.

---

### Análisis de datos

Adicionalmente, se buscó entender la relación entre las variables del conjunto de datos al graficar diferentes distribuciones y correlaciones de las mismas.

---

### Pairplot

<!-- ![](./plots/pairplot.png){.r-stretch .w-stretch} -->
![](./plots/reduced_pairplot.png){.r-stretch .w-stretch}

---

### Matriz de covarianza

![](./plots/covariance_matrix.svg)

---

### Distribución de géneros

![](./plots/genres_dist.svg)

---

### Distribución de variables numéricas

![](./plots/num_hist.svg)

---

### Distrubución de variables con respecto a género

![](./plots/boxplots/box_vote_average.svg)

---

## Clusterización

---

### Elección de K

Método del codo

![](./plots/k_means/error_by_k.svg)

---

### Elección de K

Análisis de la silueta

<!-- El análisis de la silueta mide la calidad del agrupamiento o clustering. Mide la distancia de separación entre los clústers. Nos indica como de cerca está cada punto de un clúster a puntos de los clústers vecinos. Esta medida de distancia se encuentra en el rango [-1, 1]. Un valor alto indica un buen clustering.

Los coeficientes de silueta cercanos a +1 indican que la observación se encuentra lejos de los clústers vecinos. Un valor del coeficiente de 0 indica que la observación está muy cerca o en la frontera de decisión entre dos clústers. Valores negativos indican que esas muestras quizás estén asignadas al clúster erróneo.

El método de la silueta calcula la media de los coeficientes de silueta de todas las observaciones para diferentes valores de k. El número óptimo de clústers k es aquel que maximiza la media de los coeficientes de silueta para un rango de valores de k. -->

![](./plots/k_means/silhouette.svg)

---

## Clasificación

---

### K = 3

![](./plots/k_means/classification_3.svg)

---

### K = 5

![](./plots/k_means/classification_5.svg)

---

### K = 10

![](./plots/k_means/classification_10.svg)

---

## Resultados

---

### K = 3

![](./plots/k_means/confusion_3.svg)

---

### K = 5

![](./plots/k_means/confusion_5.svg)

---

### K = 10

![](./plots/k_means/confusion_10.svg)

---

## Agrupamiento jerárquico

Este algoritmo organiza los datos en una jerarquía de clústeres, basándose en su similitud o distancia. Esta representación jerárquica se representa mediante un gráfico llamado dendograma.

![](./plots/dendogram.png)

---

### Agrupamiento

Hay dos variantes para agrupar los datos en clústeres: método aglomerativo y divisivo.

En el caso del método aglomerativo, se comienza con un grupo por cada observación. Luego, se va iterando por los clústeres tomando la distancia entre los mismos y uniendo a los clústeres con la menor distancia. La iteración termina cuando quede un único grupo.

---

### Medidas de similitud

Para medir la similitud entre dos clusters (o su "distancia"), existen varios criterios:

:::{.container .r-stretch}
::::{.flex-1}
1. Similitud máxima
    $$ 
      d_{12}=\underset{i,j}\max d(X_i,Y_j)
    $$
2. Similitud mínima
    $$ 
      d_{12}=\underset{i,j}\min d(X_i,Y_j)
    $$
::::
::::{.flex-1}
3. Similitud por centroide
    $$
      d_{12}=d(\overline x,\overline y)
    $$
4. Similitud promedio
    $$
      d_{12}={1\over kl}\sum_{i=1}^k\sum_{j=1}^l d(X_i,Y_j)
    $$
::::
:::

Donde $d$ es la función de distancia elegida.

---

## Kohonen

Las redes de Kohonen son redes capaces de descubrir por sí mismas regularidades en los datos de entrada, sin necesidad de un supervisor externo. Esta técnica sirve para generar una representación bidimensional de un espacio de datos de mayor dimensionalidad, preservando su estructura topológica.
<!-- Es decir que se preservan las relaciones espaciales entre los elemenots -->

Adicionalmente, tienen la ventaja de que ante datos similares en la entrada, siempre se generan datos similares en la salida.

---

### Kohonen - Estructura

Una red de Kohonen es un tipo de red neuronal compuesta de una única capa, llamada capa de salida, la cual tiene formato de grilla bidimensional de dimensión $k\times k$.

Cada neurona en esta red está conectada con todas las entradas. Estas conexiones se representan con un vector n-dimensional (donde $n$ es la cantidad de entradas) de pesos (también llamado prototipo). Formalmente, se tiene:

$$
w=(x_1,x_2,\dots,x_n)
$$

Adicionalmente, cada una de las neuronas se conecta tanto consigo misma (retroalimentación), como con todas sus neuronas _vecinas_.

---

### Kohonen - Aprendizaje

El método de aprendizaje empleado en una red de Kohonen se denomina **Aprendizaje competitivo**.

Esto se debe a que el sistema produce que algunas neuronas tengan mas activacion que otras en el output. Se define como neurona ganadora a la neurona que tenga vector de pesos mas similar a la entrada, quedando el resto de neuronas con valores de respuesta mínimos.

---

### Kohonen - Vecindario

Sea $(i,j)$ la neurona que se encuentra en la fila $i$, columna $j$ de la red.

Dada la neurona $(i,j)$, se define como **vecindario** de $(i,j)$ a todas las neuronas que están a una distancia menor o igual a una distancia $R$, denominada _radio del vecindario_.

$$
V_{(i,j)}=\{(i',j')\ /\ d(\ (i,j),(i',j') \le R\ )\}
$$

---

### Kohonen - Vecindario

La forma de calcular el vecindario de una neurona dependerá tanto del formato de la grilla elegido para la red (rectangular o hexagonal), como de la función de distancia que se utilice (normalmente distancia euclídea, pero puede ser de cualquier otro tipo).

Normalmente, el radio del vecindario se inicializa del tamaño de la red ($R_i=k$) y se va reduciendo en cada iteración hasta llegar a 0.

---

### Kohonen - Entrenamiento

Sea un conjunto de entrenamiento X que tiene P ejemplos, con cada ejemplo de dimensión n

$$
X=\{x^1,\dots,x^P\}\\ X^p=(x_1^p,x_2^p,\dots,x_n^p)
$$

Sea una capa de salida formada por $(k\times k)$ neuronas

Para cada neurona de la capa de salida $(i,j)$ habrá un vector de pesos de dimensión n que representa sus conexiones con la entrada

$$
w^{ij}=(w_1^{ij},w_2^{ij},\dots,w_n^{ij})\ \ \ i,j\in\{1,\dots,k\}
$$

---

### Kohonen - Entrenamiento

Para cada $X^p$ seleccionaremos una neurona ganadora. Entonces, la neurona $(i,j)$ será la ganadora si el vector de pesos $w^{ij}$ es el más parecido a la entrada $X^p$.

Luego, corregiremos el vector de pesos de la neurona ganadora para aumentar su similitud con el input $X^p$. Además, aplicaremos el mismo tipo de corrección en menor medida a todas las neuronas del vecindario de $(i,j)$, reduciendo la magnitud de la corrección según la distancia de la neurona vecina.

---

### Kohonen - Entrenamiento

De esta forma, las neuronas vecinas se irán asemejando entre sí. Este proceso se denomina **ordenamiento**.

A medida que el radio R va disminuyendo, el ordenamiento va estabilizándose y el entrenamiento irá **convergiendo**.

---

### Kohonen - Actualización de pesos

Luego de cada iteración, se acutalizarán los pesos del vector $w$, tanto para cada neurona como para su vecindario.

Sea $(i,j)$ una neurona, para cada neurona de su vecindario (incluyendo a $(i,j)$) se actualizará su vector de pesos $w_k^i,j$ tal que:

$$
w_k^{ij}(t+1)=w_k^{ij}(t)+\Delta w_k^{ij}\\
\Delta w_k^{ij} = V*\eta\ *(x_k^p-w_k^{ij})\\
V=e^{-{2d / R}},\\ d=\text{distancia con la neurona ganadora}\\
\eta=\eta_\text{inicial}-\text{cte} * k\\
R=(\text{max\_ctd\_epocas}-\text{epoca})*{R_\text{inicial}\over\text{max\_ctd\_epocas}}
$$

---

### Kohonen - Visualización

Para visualizar los resultados de este algoritmo, lo más natural es representarlos en una grilla bidimensional.

Hay dos formas principales de representación: 

- La primera consiste en graficar (en general o por variable) la cantidad de activaciones que tuvo cada neurona en la red
- En la segunda, llamada Matriz U (Unified Distance Matrix), se grafica para cada neurona el promedio de la distancia euclídea entre el vector $w_{ij}$ de pesos de la neurona, y los vectores de pesos $w_{i'j'}$ de las neuronas vecinas.
 
---

# GRACIAS
