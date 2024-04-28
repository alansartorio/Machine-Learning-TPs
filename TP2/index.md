---
title: TP1
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

Para la clasificacion primero dividimos el conjunto de datos en:
- training: ????%
- test: ????%

Lo hicimos ALEATORIAMENTE??? LA CONSIGNA DICE ALEATORIO.

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

PONER ARBOLITO

---

Ahora podemos responder la pregunta. 
CON QUE PERSONA HAY QUE TESTEAR ESTO??

---

#### Random Forest

Ademas de probarlo con un solo arbol, tambien observamos que resultado obteniamos al utilizar ??? arboles y sumando sus resultados.

Obteniendo= ?????

---

## Analisis del modelo

Finalmente para saber que tan bueno es el modelo entrenado, se realizo:
- La matriz de confusion  para un arbol y para el random forest.
- Grafico de curvas de precision del arbol teniendo en cuenta la cantidad de nodos, donde en nuestro caso, usamos la altura del arbol.


---

### Matriz de confusion

---

---

### Curvas de precision

---

# Ejercicio 2

Clasificacion con KNN
---
