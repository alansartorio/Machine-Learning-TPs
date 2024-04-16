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

---

## Datos

En este ejercicio contamos con un dataset que tiene titulares de diferentes noticias publicadas en el país. Cada documento cuenta con: 

- El texto de la noticia
- La fuente de la noticia 
- Su categoría correspondiente

---

### Análisis previo

Al analizar los datos, primero buscamos qué categorías existen dentro del dataset, ante lo cuál obtuvimos que son las 9 siguientes:

- Noticias destacadas  
- Nacional
- Destacadas
- Ciencia y Tecnologia
- Deportes
- Entretenimiento
- Economia
- Internacional
- Salud
<!-- completar -->

---

### Análisis previo

Luego, decidimos investigar la cantidad de registros por cada una de estas categorías:

|Categoría|Cantidad| 
----------|--------
Noticias destacadas |      133,819
Nacional |                 3,860
Destacadas |               3,859
Ciencia y Tecnologia |     3,856
Deportes |                 3,855
Entretenimiento |          3,850
Economia |                 3,850
Internacional |            3,850
Salud |                    3,840

---

### Análisis previo

Por este resultado decidimos recortar el dataset original para limitar la cantidad de registros por categoría. De esta forma, reducimos el dataset para que cada clase cuente con **3,840 registros**.

---

### Análisis previo

Otro aspecto a analizar del dataset es el vocabulario de los titulares, ya que será lo que usaremos para el entrenamiento del clasificador. Para esto, decidimos armar una *Bag of Words* de las palabras que aparecen en cada titular. Previamente a esto hicimos una pre-tokenización de los titulares, removiendo signos de puntuación y convirtiendo todas las palabras a minúsculas para evitar diferenciación por la capitalización. 

---

### Análisis previo

<div style="display: flex; flex-direction: row; width:100%;align-items:center;gap:2rem">
![](./plots/word_count.svg)
<p style="max-width: 40%;font-size:1.5rem;text-align:left">
Al graficar las primeras 30 palabras ordenadas por la cantidad de apariciones en el vocabulario, podemos observar la presencia de varias *stop words*, cuya frecuencia de aparición es significativamente mayor al resto.
Por esto, decidimos tomar como *stop words* las 20 palabras más frecuentes que aquí figuran y **removerlas del dataset**.
</p>


---

### Transformación del dataset

Para poder implementar un clasificador de texto en base al dataset provisto, primero necesitamos transformar los datos de entrada para que puedan ser procesados por el algoritmo de Naive Bayes.

Para esto, decidimos transformar los titulares en *"vectores de vocabulario"*. Estos vectores tienen la misma dimensión que el tamaño de nuestro vocabulario, donde cada posición representa una palabra del mismo. De esta forma, cada titular tendrá un **1** en las posiciones correspondientes a las palabras incluídas en el mismo, y un **0** en el resto. El vocabulario consiste en un set de todas las plabras incluídas en los titulares.

---

### Transformación del dataset

Finalmente, para el entrenamiento del algoritmo dividimos el dataset en dos: uno de **training** y uno de **test**. 

Para evitar que estas particiones queden desbalanceadas nos aseguramos de que ambas cuenten con la misma proporción de documentos para cada categoría.

---

## Clasificación

Al tener la división propuesta de nuestro dataset, usaremos los datos de training para calcular las probabilidades condicionales de cada palabra dentro del vocabulario con respecto a cada categoría, además de las probabilidades de ocurrencia de cada una de las categorías y de los titulares.

Luego, utilizaremos esos datos para estimar la categoría a la que pertenece cada uno de los titulares del dataset de test, pudiendo comparar luego con el valor real.

---

### Cálculo de probabilidades

Para estimar la categoría de un titular utilizaremos el clasificador de *Naive Bayes*. Tomaremos como datos a los titulares, a los cuáles consideramos como un vector de valores binarios

$T = (w_1, w_2, \dots, w_n)$

Tal que $w_i = 1$ si la i-esima palabra del vocabulario está en $T$ y $w_i = 0$ en caso contrario.

Entonces, utilizamos el teorema de Bayes para calcular la probabilidad de que el titular pertenezca a cada una de las categorías dadas las palabras que contiene.

$P(c|T) =  {P(c) P(T|c)\over P(T)}$

---

### Cálculo de probabilidades

Similar al ejercicio anterior, utilizaremos el hecho de que se asumen las variables (en nuestro caso, cada una de las palabras) como independientes con respecto a la clase para calcular la probabilidad condicional del numerador

$P(T|c) = \prod_{w_i\in T} P(w_i|c)$

Asimismo, utilizamos probabilidad total para el cálculo de la probabilidad del titular.

$P(T) = \sum_{c_i\in C} [P(c_i) * \prod_{w_i\in T} P(w_i|c_i)]$

---

## Matriz de confusión

![](./plots/2_confusion_matrix.svg)

---

## Metricas

![](./plots/2_metrics.svg)

<!-- Si el F1 sigue dando mal, podríamos agregar una hipótesis de por qué está así -->
<!-- En general se podría añadir un análisis de los datos -->

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
