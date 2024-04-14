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

Clasificación de nacionalidad de personas sabiendo sus preferencias

---

## Datos
Para este ejercicio se nos dio un dataset en el cuál se registran la nacionalidad (Escocés o Inglés) de diferentes personas y si les gustan o no los siguientes atributos:

- scones
- cerveza
- wiskey
- avena
- futbol


---

### Análisis previo

Para entender mejor los datos decidimos analizar la proporción de gente, agrupada por nacionalidad, que prefiere cada uno de los atributos

<div style="display: flex; flex-direction: row; width:100%;align-items:center;gap:2rem">
![](./plots/relative_preference.svg){max-width=50%}

<p style="max-width: 40%;font-size:1.5rem;text-align:justify">
En base a este resultado se puede observar que, mientras las preferencias de los ingleses están distribuidos de forma bastante uniforme (no hay ningún atributo que sea preferido por más de la mitad de la población), en el caso de los escoceses hay una gran preferencia tanto por los scones como por la avena.
</p> 
</div>


---

## Clasificación

En base a estos datos, se nos pide que determinemos la nacionalidad de dos personas en base a sus preferencias.

Para esto, utilizaremos el algoritmo de Naive Bayes con el cuál podremos clasificar a las personas en base a la nacionalidad que resulte con valor máximo. 

---

### Cálculo de probabilidades

<p style="font-size:1.5rem">
Podremos obtener valores de probabilidades al dividir los resultados de Naive Bayes por la probabilidad del dato de la persona. Este último lo obtenemos mediante el cálculo de **Probabilidad Total**:
</p>

$P(D) = \sum_{c_i\in C}\ {P(D |c_i) * P(c_i)}$

<p style="font-size:1.5rem">
En este caso, asumimos que las variables son independientes dada una clase, por lo que la ecuación nos queda como:
</p>

$$
P(D/C) = \prod_{d_i\in D}\ P(d_i|C)\ \implies \ P(D) = \sum_{c_i\in C}\ {(\prod_{d_i\in D}\ P(d_i|c_i) * P(c_i))}
$$



<!-- Acá se puede explicar brevemente cómo lo hacemos -->

---

## Clasificar el ejemplo 

### $x_1 = (1, 0, 1, 1, 0)$

En este caso tratamos con una persona a la que:

- <span style="color:green">Le gustan los scones </span>
- <span style="color:red">No le gusta la cerveza</span>
- <span style="color:green">Le gusta el whisky</span>
- <span style="color:green">Le gusta la avena</span>
- <span style="color:red">No juega al fútbol</span>

---

### Resultado
<!--- Mostrar lo que supuestamente vimos a ojo en los datos del input (algun plot supongo) -->
<!--- Mostar los nombres de cada variable asignados a sus valores -->

Para el primer ejemplo, al correr el clasificados obtuvimos que la persona es **Escocesa**. Para obtener más detalle sobre el resultado, calculamos las probabilidades para cada nacionalidad.

![](./plots/x1.svg)

---

## Clasificar el ejemplo $x_2 = (0, 1, 1, 0, 1)$

En este caso tratamos con una persona a la que:

- <span style="color:red">No le gustan los scones</span>
- <span style="color:green">Le gusta la cerveza</span>
- <span style="color:green">Le gusta el whisky</span>
- <span style="color:red">No le gusta la avena</span>
- <span style="color:green">Juega al fútbol</span>

---

### Resultado

Esta vez, obtuvimos como resultado que la persona en cuestión es **Inglesa**. Luego, al calcular las probabilidades obtenemos:

![](./plots/x2.svg)



# Ejercicio 2

Clasificar noticias argentinas

<!--- Explicar como tokenizamos los titulos en una diapo mas -->

---

## Datos de entrenamiento y prueba

<!--- Por ahí correr de nuevo reduciendo la categoria con 100 mil noticias -->
<!--- Decir porcentaje -->

Dividimos el conjunto de datos agrupando por categoria de noticia y tomando un porcentaje de cada grupo para entrenamiento y lo que queda para prueba.

---

## Matriz de confusión

![](./plots/2_confusion_matrix.svg)

---

## Metricas

![](./plots/2_metrics.svg)

---

## Curva ROC

<!--- Hacer de nuevo curva ROC variando el threshold de cada clase -->

![](./plots/2_roc_curve.svg)




# Ejercicio 3

Admisión de estudiantes

---

## Probabilidad de admisión, caso A

<!--- Mencionar si usamos la correccion de laplace (o sacarla por no ser necesaria) -->
<!--- Corregir la corrección de laplace porque el classes_amount parecia estar mal usado? (si no la usamos no deberia ser necesario) -->
<!--- Agregar que significa caso A (consigna e input) -->

![](./plots/3_a.svg)

---

## Probabilidad de admisión, caso B

<!--- Agregar que significa caso B (consigna e input) -->

![](./plots/3_b.svg)

---

## Proceso de aprendizaje

El proceso de aprendizaje en este caso se divide en dos partes principales: primero, investigamos cómo están conectadas entre sí las diferentes variables, es decir, cuál influye en cuál. Esto lo hacemos explorando todas las conexiones posibles y eligiendo las que mejor explican los datos que tenemos. Una vez que sabemos cómo se relacionan las variables, el siguiente paso es calcular qué tan fuerte es cada una de estas conexiones, utilizando los datos para calcular probabilidades que nos muestren esta intensidad.

<!--- Agregar conclusiones -->
