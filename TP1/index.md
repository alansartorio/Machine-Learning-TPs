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

## Clasificar el ejemplo $x_1 = (1, 0, 1, 1, 0)$

![](./plots/x1.svg)

---

## Clasificar el ejemplo $x_2 = (0, 1, 1, 0, 1)$

![](./plots/x2.svg)



# Ejercicio 2

Clasificar noticias argentinas

---

## Datos de entrenamiento y prueba

Dividimos el conjunto de datos agrupando por categoria de noticia y tomando un porcentaje de cada grupo para entrenamiento y lo que queda para prueba

---

## Matriz de confusión

![](./plots/2_confusion_matrix.svg)

---

## Metricas

![](./plots/2_metrics.svg)

---

## Curva ROC

![](./plots/2_roc_curve.svg)




# Ejercicio 3

Admisión de estudiantes

---

## Probabilidad de admisión, caso A

![](./plots/3_a.svg)

---

## Probabilidad de admisión, caso B

![](./plots/3_b.svg)

---

## Proceso de aprendizaje

El proceso de aprendizaje en este caso se divide en dos partes principales: primero, investigamos cómo están conectadas entre sí las diferentes variables, es decir, cuál influye en cuál. Esto lo hacemos explorando todas las conexiones posibles y eligiendo las que mejor explican los datos que tenemos. Una vez que sabemos cómo se relacionan las variables, el siguiente paso es calcular qué tan fuerte es cada una de estas conexiones, utilizando los datos para calcular probabilidades que nos muestren esta intensidad.
