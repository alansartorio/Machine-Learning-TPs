---
title: Ejercicio Datos Alimenticios
# subtitle: Aprendizaje Automático
author:
- Benvenuto Agustín - 61448
- Galarza Agustín - 61481
- Sartorio Alan - 61379
# date:
# - 13 de Marzo 2024
# theme: "Copenhagen"
---

## Parte 1

En los datos provistos nos encontramos con algunas irregularidades representados con 999.99. En nuestro caso, decidimos eliminar estas filas ya que la irregularidad del dato hace que no sea util para analisis.

---

## Parte 2

![](./part_2.svg)

---

## Parte 2

Mirando el grafico de cada variable podemos notar algunas caracteristicas particulares:

- Las grasas saturadas tienen varios valores atipicos maximos y una distribucion de los datos bastante homogenea.
- Con el alcohol nuevamente nos encontramos con varios valores atipicos maximos, pero en este caso contamos con pocos valores por debajo del RIC.
- Sobre las calorias, se muestra un grafico relativamente similar a las grasas saturadas.
- En a los sexos, podemos ver que la cantidad de datos provista para cada sexo es suficiente para este tipo de analisis (+30 muestras)

---

## Parte 3

![](./part_3.svg)

---

## Parte 3

Haciendo la distincion por sexo podemos notar:

- Las calorias y grasas saturadas varian muy poco al cambiar el sexo, pero si se notan algunos casos maximos atipicos en las mujeres.
- El caso del alcohol muetra un mayor rango y media en las mujeres.

---

## Parte 4

![](./part_4.svg)

---

## Parte 4

Como se puede ver hay una clara tendencia creciente. A mayor cantidad de calorias consumidas, mayor cantidad de alcohol.
Por otro lado, se notan drasticas diferencias entre los diferentes sexos que va creciendo junto con la cantidad de calorias. El caso mas notorio es que el valor atipico maximo de las mujeres es comparable al valor mínimo para los hombres en el caso de calorias [0, 1100]. De la misma manera, la media es muchisimo mayor (casi el doble) y los valores atipicos maximos pasan a numeros de caso el triple de la media de los hombres.
