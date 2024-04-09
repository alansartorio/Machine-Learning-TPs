Tenemos:

Clases: $\{\text{estudiante} (E), \text{graduado} (G)\}$

Programas: $\{A, B, C, D\}$

Donde:

$$
\begin{aligned}
P(A/E) &= 0.95 \\
P(B/E) &= 0.05 \\
P(C/E) &= 0.02 \\
P(D/E) &= 0.2
\end{aligned}
$$

Y para los graduados:

$$
\begin{aligned}
P(A/G) &= 0.03 \\
P(B/G) &= 0.82 \\
P(C/G) &= 0.34 \\
P(D/G) &= 0.92
\end{aligned}
$$

Finalmente, tambien sabemos que:

$$
\begin{aligned}
P(G) &= 0.9 \\
P(E) &= P(\bar{G}) \\
&= 1 - P(G) \\
&= 0.1 \\
\end{aligned}
$$

**Queremos encontrar**:
Un nuevo oyente escucha los programas A y C pero no le gustan los programasB y D. Calcular la probabilidad de que este oyente sea estudiante y la probabilidadde que sea graduado.

Es decir:

$P(G/A,\bar{B},C,\bar{D})$
y
$P(E/A,\bar{B},C,\bar{D})$

Comenzamos por buscar la probabilidad condicional pero invertida:

Asumiendo que las variables son independientes:

$$
\begin{aligned}
P(A,\bar{B},C,\bar{D}/E) &= P(A/E) * P(\bar{B}/E) * P(C/E) * P(\bar{D}/E) \\
&= 0.95 * (1 - 0.05) * 0.02 * (1 - 0.2) \\
&= 0.01444
\end{aligned}
$$

$$
\begin{aligned}
P(A,\bar{B},C,\bar{D}/G) &= P(A/G) * P(\bar{B}/G) * P(C/G) * P(\bar{D}/G) \\
&= 0.03 * (1-0.82) * 0.34 * (1-0.92) \\
&= 0.00014688
\end{aligned}
$$

Luego, mediante Bayes, sabemos que:

$$
\begin{aligned}
P(E/A,\bar{B},C,\bar{D})=\frac{P(A,\bar{B},C,\bar{D}/E) * P(E)}{P(A,\bar{B},C,\bar{D})}
\end{aligned}
$$

Ídem para los graduados (G)

Donde:

$$
P(E)= 0.1 \space (dato)
$$
$$
P(A,\bar{B},C,\bar{D}/E)\space (calculado)
$$

Solo falta:

$$
\begin{aligned}
P(A,\bar{B},C,\bar{D})&=P(A,\bar{B},C,\bar{D}/E) \times P(E) + P(A,\bar{B},C,\bar{D}/G) \times P(G) \\
&= 0.00014688 \times 0.1 + 0.01444 \times 0.9 \\
&= 0.001572192
\end{aligned}
$$

Entonces:

$$
\begin{aligned}
P(G/A,\bar{B},C,\bar{D}) &= \frac{0.9 \times 0.0014688}{0.001572192}\\
P(G/A,\bar{B},C,\bar{D}) &= 0.08408\\
P(E/A,\bar{B},C,\bar{D}) &= \frac{0.1 \times 0.01444}{0.001572192}\\
P(E/A,\bar{B},C,\bar{D}) &= 0.91462
\end{aligned}
$$
