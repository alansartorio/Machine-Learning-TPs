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

![](./plots/pairplot.svg)

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

## Clasificación

---

## Image

![](./plots/k_means/error_by_k.svg){.r-stretch}

---

![](./plots/k_means/silhouette.svg){.r-stretch}

---

![](./plots/k_means/classification.svg){.r-stretch}

---

## Side to side

:::{.container .r-stretch}
::::{.flex-1}
![](./plots/.gif)
::::
::::{.flex-1}
![](./plots/.svg)
::::
:::

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



---

---

# GRACIAS
