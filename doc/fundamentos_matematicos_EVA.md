# Fundamentos Matemáticos

**Sistema inteligente para calcular las tensiones mecánicas de un rostro robótico mediante algoritmos genéticos para la imitación de expresiones faciales humanas, basado en el caso EVA**

---

## Índice

1. [Formulación Matemática del Problema](#1-formulación-matemática-del-problema)
2. [Sistema de Codificación Binaria](#2-sistema-de-codificación-binaria)
3. [Generación de la Población Inicial](#3-generación-de-la-población-inicial)
4. [Función de Aptitud (Fitness)](#4-función-de-aptitud-fitness)
5. [Selección por Torneo](#5-selección-por-torneo)
6. [Cruzamiento (Crossover)](#6-cruzamiento-crossover)
7. [Mutación](#7-mutación)
8. [Reemplazo con Elitismo](#8-reemplazo-con-elitismo)
9. [Poda de la Población](#9-poda-de-la-población)
10. [Condición de Término](#10-condición-de-término)
11. [Métricas de Estabilidad y Diversidad](#11-métricas-de-estabilidad-y-diversidad)
12. [Validación Cruzada (Cross-Validation)](#12-validación-cruzada-cross-validation)
13. [Flujo Completo del Sistema](#13-flujo-completo-del-sistema)

---

## 1. Formulación Matemática del Problema

### 1.1 Definición del Espacio de Búsqueda

El problema consiste en encontrar un vector de tensiones mecánicas $M$ para $n = 6$ actuadores del rostro robótico, tal que la expresión facial resultante sea lo más cercana posible a la expresión del usuario humano capturada por cámara.

**Vector de tensiones (incógnita):**

$$M = [m_1, m_2, m_3, m_4, m_5, m_6]$$

donde cada $m_i \in [0.0, 1.0]$ representa la tensión normalizada del actuador $i$.

**Vector objetivo (dato de entrada):**

$$T = [t_1, t_2, t_3, t_4, t_5, t_6]$$

donde cada $t_i \in [0.0, 1.0]$ es el valor del blendshape facial correspondiente extraído por MediaPipe.

### 1.2 Mapeo de Actuadores a Blendshapes

Cada componente del vector $T$ se obtiene a partir de los coeficientes de expresión facial (blendshapes) detectados por MediaPipe, según la siguiente correspondencia:

| Componente | Blendshape de origen | Transformación                     | Actuador                |
| ---------- | -------------------- | ---------------------------------- | ----------------------- |
| $t_1$      | `jawOpen`            | $t_1 = \text{jawOpen}$             | $m_1$ — Mandibular      |
| $t_2$      | `mouthSmileLeft`     | $t_2 = \text{mouthSmileLeft}$      | $m_2$ — Cigomático izq. |
| $t_3$      | `mouthSmileRight`    | $t_3 = \text{mouthSmileRight}$     | $m_3$ — Cigomático der. |
| $t_4$      | `eyeBlinkLeft`       | $t_4 = 1.0 - \text{eyeBlinkLeft}$  | $m_4$ — Orbicular izq.  |
| $t_5$      | `eyeBlinkRight`      | $t_5 = 1.0 - \text{eyeBlinkRight}$ | $m_5$ — Orbicular der.  |
| $t_6$      | `browDownLeft`       | $t_6 = 1.0 - \text{browDownLeft}$  | $m_6$ — Frontal         |

> **Nota:** La inversión ($1.0 - x$) en $t_4$, $t_5$ y $t_6$ se debe a que MediaPipe reporta el "grado de cierre" (pestañeo/fruncimiento), pero el actuador robótico modela el "grado de apertura". Invertir el valor garantiza la coherencia semántica.

### 1.3 Objetivo de Optimización

El Algoritmo Genético busca encontrar el vector $M^*$ que minimice el error cuadrático medio entre el vector objetivo $T$ y la respuesta del gemelo digital ante las tensiones $M$:

$$M^* = \arg\min_{M \in [0,1]^6} E(M, T)$$

donde $E$ es el Error Cuadrático Medio (MSE) definido formalmente en la Sección 4.

---

## 2. Sistema de Codificación Binaria

### 2.1 ¿Por qué codificación binaria?

El Algoritmo Genético no manipula directamente los valores reales $m_i$. En un AG canónico (estándar académico), cada variable de decisión se representa como una **cadena de bits** (unos y ceros). Esto permite aplicar operadores genéticos clásicos (cruza de punto, mutación de bit) y garantiza un espacio de búsqueda discreto y uniforme.

### 2.2 Parámetros del Sistema de Codificación

Para cada actuador $m_i$, debemos definir los parámetros que gobiernan la conversión entre el mundo binario y el mundo real.

**Datos de entrada:**

- Límite inferior del rango: $a = 0.0$
- Límite superior del rango: $b = 1.0$
- Resolución deseada: $\delta = 10^{-3}$ (precisión de milésimas)

**Cálculo del rango:**

$$R = b - a = 1.0 - 0.0 = 1.0$$

**Puntos necesarios para cubrir el rango con la resolución deseada:**

$$P_{problema} = \left\lfloor \frac{R}{\delta} \right\rfloor + 1 = \left\lfloor \frac{1.0}{0.001} \right\rfloor + 1 = 1001$$

**Número de bits necesarios por actuador:**

$$k = \lceil \log_2(P_{problema}) \rceil = \lceil \log_2(1001) \rceil = \lceil 9.967 \rceil = 10 \text{ bits}$$

**Puntos que el sistema binario realmente puede representar:**

$$P_{sistema} = 2^k = 2^{10} = 1024$$

**Resolución real del sistema (ligeramente mejor que la deseada):**

$$\delta_{real} = \frac{R}{P_{sistema} - 1} = \frac{1.0}{1023} \approx 0.000978$$

### 2.3 Estructura del Cromosoma Completo

Dado que tenemos $n = 6$ actuadores y cada uno requiere $k = 10$ bits, el cromosoma completo de un individuo tiene una longitud total de:

$$L = n \times k = 6 \times 10 = 60 \text{ bits}$$

La estructura del cromosoma es:

```
|--- m₁ ---|--- m₂ ---|--- m₃ ---|--- m₄ ---|--- m₅ ---|--- m₆ ---|
| b₁...b₁₀ | b₁₁..b₂₀| b₂₁..b₃₀| b₃₁..b₄₀| b₄₁..b₅₀| b₅₁..b₆₀|
```

**Ejemplo concreto:**

```
Cromosoma: 1101100011 0111010010 1010101010 1100000110 0110011001 1000011101
           |-- m₁ --| |-- m₂ --| |-- m₃ --| |-- m₄ --| |-- m₅ --| |-- m₆ --|
```

### 2.4 Decodificación: De Bits a Valor Real

Para extraer el valor real $m_i$ a partir de su segmento de bits, se aplica el siguiente procedimiento:

**Paso 1 — Extraer el segmento de bits del actuador $i$:**

$$\text{bits}_i = \text{cromosoma}[(i-1) \cdot k : i \cdot k]$$

**Paso 2 — Convertir la cadena binaria a valor decimal:**

$$d_i = \sum_{j=0}^{k-1} b_j \cdot 2^{(k-1-j)}$$

donde $b_j$ es el bit en la posición $j$ de la cadena (de izquierda a derecha).

**Paso 3 — Mapear el valor decimal al rango real $[a, b]$:**

$$m_i = a + d_i \cdot \frac{R}{2^k - 1} = 0.0 + d_i \cdot \frac{1.0}{1023}$$

**Ejemplo numérico completo:**

```
Segmento de bits para m₁: 1101100011

Paso 2: d₁ = 1·2⁹ + 1·2⁸ + 0·2⁷ + 1·2⁶ + 1·2⁵ + 0·2⁴ + 0·2³ + 0·2² + 1·2¹ + 1·2⁰
       d₁ = 512 + 256 + 0 + 64 + 32 + 0 + 0 + 0 + 2 + 1 = 867

Paso 3: m₁ = 0.0 + 867 · (1.0 / 1023) = 0.84750
```

Esto significa que el actuador mandibular (apertura de boca) debe aplicar una tensión del **84.75%** de su capacidad máxima.

---

## 3. Generación de la Población Inicial

### 3.1 Proceso de Generación

La población inicial $P_0$ se compone de $N$ individuos generados de forma completamente aleatoria. Para cada individuo $j$:

$$P_0 = \{I_1, I_2, ..., I_N\}$$

donde cada individuo $I_j$ es un cromosoma de $L = 60$ bits generado de manera uniformemente aleatoria:

$$I_j = [b_1, b_2, ..., b_{60}], \quad b_l \sim \text{Bernoulli}(0.5) \quad \forall l \in \{1, ..., 60\}$$

Es decir, cada bit tiene un 50% de probabilidad de ser `0` y un 50% de ser `1`.

### 3.2 Evaluación Inicial

Inmediatamente después de generar cada individuo, se decodifica su cromosoma (Sección 2.4) para obtener su vector de tensiones $M_j = [m_1^j, ..., m_6^j]$ y se evalúa su aptitud mediante la función de fitness (Sección 4).

### 3.3 Parámetros Configurables

| Parámetro              | Símbolo         | Rango Recomendado | Valor Por Defecto |
| ---------------------- | --------------- | ----------------- | ----------------- |
| Tamaño de población    | $N$             | 30 – 200          | 50                |
| Bits por actuador      | $k$             | 8 – 16            | 10                |
| Número de actuadores   | $n$             | 6                 | 6                 |
| Longitud del cromosoma | $L = n \cdot k$ | 48 – 96           | 60                |

---

## 4. Función de Aptitud (Fitness)

### 4.1 Error Cuadrático Medio (MSE)

El Error Cuadrático Medio mide la distancia euclidiana promedio entre el vector de tensiones del individuo ($M$) y el vector objetivo ($T$):

$$E(M, T) = \frac{1}{n} \sum_{i=1}^{n} (t_i - m_i)^2$$

donde:

- $n = 6$ es el número de actuadores
- $t_i$ es el valor objetivo del actuador $i$ (del usuario humano)
- $m_i$ es el valor decodificado del actuador $i$ (del cromosoma)

**Propiedades del MSE:**

- $E \geq 0$ siempre (la suma de cuadrados nunca es negativa)
- $E = 0$ si y solo si $m_i = t_i$ para todo $i$ (imitación perfecta)
- $E_{max} = 1.0$ cuando todos los actuadores están en el extremo opuesto ($m_i = 0$ cuando $t_i = 1$ o viceversa)

### 4.2 Función de Aptitud (Fitness)

Dado que el AG busca **maximizar** la aptitud, pero el error $E$ debe **minimizarse**, se aplica una transformación inversa:

$$F(M, T) = \frac{1}{1 + E(M, T)}$$

**Propiedades de la función de aptitud:**

- $F \in (0, 1]$ (siempre positiva, máximo valor es 1)
- $F = 1$ cuando $E = 0$ (imitación perfecta → aptitud máxima)
- $F \to 0$ cuando $E \to \infty$ (error enorme → aptitud nula)
- La función es **monótonamente decreciente** respecto a $E$: a menor error, mayor aptitud

**Ejemplo numérico:**

```
Vector objetivo (del humano):   T = [0.80, 0.90, 0.85, 0.70, 0.75, 0.60]
Vector del individuo (del AG):  M = [0.78, 0.87, 0.83, 0.65, 0.72, 0.55]

Cálculo del MSE:
E = (1/6)·[(0.80-0.78)² + (0.90-0.87)² + (0.85-0.83)² + (0.70-0.65)² + (0.75-0.72)² + (0.60-0.55)²]
E = (1/6)·[0.0004 + 0.0009 + 0.0004 + 0.0025 + 0.0009 + 0.0025]
E = (1/6)·[0.0076]
E = 0.001267

Cálculo del Fitness:
F = 1 / (1 + 0.001267) = 0.99874

Interpretación: Este individuo imita la cara del usuario con un 99.87% de aptitud.
```

### 4.3 Gráfica Conceptual de la Función de Aptitud

```
F(E)
 1.0 ┤●
     │ ╲
 0.8 ┤  ╲
     │   ╲
 0.6 ┤    ╲
     │      ╲
 0.4 ┤        ╲
     │          ╲___
 0.2 ┤              ╲________
     │                        ╲_______________
 0.0 ┤─────────────────────────────────────────→ E
     0    0.5    1.0    1.5    2.0    2.5    3.0
```

La curva muestra el carácter asintótico de $F$: los individuos con errores pequeños reciben aptitudes muy altas, pero a medida que el error crece, la aptitud se reduce rápidamente y luego se aplana (rendimientos decrecientes del castigo).

---

## 5. Selección por Torneo

### 5.1 Descripción del Mecanismo

La selección por torneo es el método mediante el cual el AG elige qué individuos serán los "padres" de la siguiente generación. Es preferido sobre la selección por ruleta porque mantiene una presión selectiva controlable sin requerir el cálculo de probabilidades proporcionales.

### 5.2 Algoritmo Formal

Para seleccionar un padre:

1. Elegir aleatoriamente $k$ individuos de la población actual $P_g$ (con reemplazo o sin reemplazo).
2. Comparar la aptitud $F$ de los $k$ competidores.
3. El individuo con mayor $F$ gana el torneo y es seleccionado como padre.

**Formalización:**

$$\text{Padre} = \arg\max_{I \in S_k} F(I)$$

donde $S_k \subset P_g$ es un subconjunto aleatorio de tamaño $k$.

### 5.3 Parámetro de Presión Selectiva ($k$)

| Valor de $k$          | Efecto en la evolución                                       |
| --------------------- | ------------------------------------------------------------ |
| $k = 2$               | Presión baja: más diversidad, convergencia lenta             |
| $k = 3$ (recomendado) | Presión equilibrada: buen balance exploración/explotación    |
| $k = 5$               | Presión alta: convergencia rápida, riesgo de mínimos locales |
| $k = N$               | Determinístico: siempre gana el mejor (pierde diversidad)    |

### 5.4 Ejemplo de un Torneo ($k = 3$)

```
Población de 5 individuos:
  I₁: F = 0.9821
  I₂: F = 0.8534
  I₃: F = 0.9102
  I₄: F = 0.7215
  I₅: F = 0.9456

Torneo 1: Se eligen al azar {I₂, I₄, I₅}
  Competidores: F = {0.8534, 0.7215, 0.9456}
  Ganador: I₅ con F = 0.9456 → Padre 1

Torneo 2: Se eligen al azar {I₁, I₃, I₄}
  Competidores: F = {0.9821, 0.9102, 0.7215}
  Ganador: I₁ con F = 0.9821 → Padre 2
```

### 5.5 Parámetros Configurables

| Parámetro         | Símbolo | Rango     | Valor Por Defecto |
| ----------------- | ------- | --------- | ----------------- |
| Tamaño del torneo | $k$     | 2 – 7     | 3                 |
| Tasa de selección | $r_s$   | 0.5 – 1.0 | 1.0               |

La tasa de selección $r_s$ determina qué proporción de la población participa en los torneos. Con $r_s = 1.0$, toda la población participa.

---

## 6. Cruzamiento (Crossover)

### 6.1 Cruza de Un Punto

La cruza de un punto toma dos cromosomas padres ($P_1$ y $P_2$) y genera dos hijos ($H_1$ y $H_2$) intercambiando segmentos de bits a partir de un punto de corte aleatorio.

**Paso 1 — Generar punto de corte:**

$$c \sim \text{Uniforme}\{1, 2, ..., L-1\}$$

donde $L = 60$ es la longitud del cromosoma.

**Paso 2 — Crear los hijos:**

$$H_1 = P_1[1:c] \oplus P_2[c+1:L]$$
$$H_2 = P_2[1:c] \oplus P_1[c+1:L]$$

donde $\oplus$ denota la concatenación de segmentos.

**Ejemplo visual ($L = 60$, punto de corte $c = 25$):**

```
Padre 1: 1101100011 0111010010 10101 | 01010 1100000110 0110011001 1000011101
Padre 2: 0010011100 1000101101 01010 | 10101 0011111001 1001100110 0111100010
                                      ↑ c=25

Hijo 1:  1101100011 0111010010 10101 | 10101 0011111001 1001100110 0111100010
Hijo 2:  0010011100 1000101101 01010 | 01010 1100000110 0110011001 1000011101
```

### 6.2 Cruza de Dos Puntos

Extiende el mecanismo anterior con dos puntos de corte, intercambiando el segmento central entre los padres.

**Puntos de corte:**

$$c_1 \sim \text{Uniforme}\{1, ..., L-2\}, \quad c_2 \sim \text{Uniforme}\{c_1+1, ..., L-1\}$$

**Hijos resultantes:**

$$H_1 = P_1[1:c_1] \oplus P_2[c_1+1:c_2] \oplus P_1[c_2+1:L]$$
$$H_2 = P_2[1:c_1] \oplus P_1[c_1+1:c_2] \oplus P_2[c_2+1:L]$$

**Ejemplo visual ($c_1 = 15$, $c_2 = 40$):**

```
Padre 1: 110110001101110 | 1001010101010101100000 | 01100110011000011101
Padre 2: 001001110010001 | 0110101010001111100110 | 01001100110011110001
                          ↑ c₁=15                   ↑ c₂=40

Hijo 1:  110110001101110 | 0110101010001111100110 | 01100110011000011101
Hijo 2:  001001110010001 | 1001010101010101100000 | 01001100110011110001
```

### 6.3 Probabilidad de Cruza

La cruza no siempre ocurre. Antes de cruzar cada pareja, se genera un número aleatorio $r \sim \text{Uniforme}(0, 1)$. La cruza se ejecuta si y solo si:

$$r < p_c$$

donde $p_c$ es la probabilidad de cruza. Si $r \geq p_c$, los hijos son copias idénticas de los padres.

### 6.4 Significado Biológico en el Contexto EVA

Cuando el punto de corte cae en la **frontera exacta entre dos actuadores** (posiciones 10, 20, 30, 40, 50), los hijos heredan grupos musculares completos de cada padre. Cuando el corte cae **dentro** de un segmento de actuador, se produce una recombinación fina que genera tensiones híbridas que ningún padre poseía, ampliando la exploración del espacio de soluciones.

### 6.5 Parámetros Configurables

| Parámetro             | Símbolo | Rango                   | Valor Por Defecto |
| --------------------- | ------- | ----------------------- | ----------------- |
| Probabilidad de cruza | $p_c$   | 0.5 – 1.0               | 0.8               |
| Tipo de cruza         | —       | 1 punto / 2 puntos      | 1 punto           |
| Punto(s) de cruza     | $c$     | Aleatorio en $[1, L-1]$ | Aleatorio         |

---

## 7. Mutación

### 7.1 Mutación por Inversión de Bit (Bit-Flip)

La mutación introduce variabilidad aleatoria en los cromosomas para evitar que la población quede atrapada en mínimos locales. Se recorre cada bit del cromosoma de un hijo y se invierte con una probabilidad baja.

**Para cada bit $b_l$ del cromosoma ($l = 1, ..., L$):**

$$b_l' = \begin{cases} 1 - b_l & \text{si } r_l < p_m \\ b_l & \text{si } r_l \geq p_m \end{cases}$$

donde $r_l \sim \text{Uniforme}(0, 1)$ y $p_m$ es la probabilidad de mutación por bit.

### 7.2 Número Esperado de Bits Mutados

En cada cromosoma de $L = 60$ bits, el número esperado de mutaciones por individuo es:

$$\mathbb{E}[\text{mutaciones}] = L \cdot p_m = 60 \cdot 0.01 = 0.6 \text{ bits}$$

Esto significa que, en promedio, **menos de un bit cambia por individuo por generación**, lo cual es una perturbación mínima y controlada.

### 7.3 Ejemplo de Mutación

```
Cromosoma original: 1101100011 0111010010 1010101010 1100000110 0110011001 1000011101
                                                ↑ bit 33 muta

Cromosoma mutado:   1101100011 0111010010 1010101010 1110000110 0110011001 1000011101
                                                ↑ 0→1

Efecto: El bit 33 pertenece al actuador m₄ (párpado izquierdo).
  Valor decimal original de m₄: bits "1100000110" → d = 774 → m₄ = 0.7565
  Valor decimal mutado   de m₄: bits "1110000110" → d = 902 → m₄ = 0.8817

  Resultado: El párpado izquierdo pasó de 75.65% a 88.17% de apertura.
  Un sutil cambio de un solo bit alteró una microexpresión.
```

### 7.4 Impacto Posicional del Bit Mutado

No todos los bits tienen el mismo peso. Mutar un bit de alto orden (izquierda) produce cambios drásticos, mientras que mutar un bit de bajo orden (derecha) produce ajustes finos:

| Posición del bit                  | Peso binario ($2^j$) | Cambio máximo en $m_i$                         |
| --------------------------------- | -------------------- | ---------------------------------------------- |
| Bit 1 (MSB, más significativo)    | $2^9 = 512$          | $\Delta m_i = \frac{512}{1023} \approx 0.5005$ |
| Bit 5 (medio)                     | $2^5 = 32$           | $\Delta m_i = \frac{32}{1023} \approx 0.0313$  |
| Bit 10 (LSB, menos significativo) | $2^0 = 1$            | $\Delta m_i = \frac{1}{1023} \approx 0.0010$   |

Esto otorga al AG la capacidad de hacer tanto "saltos grandes" (exploración) como "ajustes microscópicos" (explotación) de forma natural y probabilística.

### 7.5 Parámetros Configurables

| Parámetro                | Símbolo | Rango           | Valor Por Defecto |
| ------------------------ | ------- | --------------- | ----------------- |
| Probabilidad de mutación | $p_m$   | 0.001 – 0.05    | 0.01              |
| Tipo de mutación         | —       | Bit-flip / Swap | Bit-flip          |

---

## 8. Reemplazo con Elitismo

### 8.1 Concepto

El reemplazo determina cómo se conforma la nueva generación $P_{g+1}$ a partir de la generación actual $P_g$ y los hijos producidos por cruza y mutación. El elitismo garantiza que los mejores individuos nunca se pierdan entre generaciones.

### 8.2 Algoritmo de Reemplazo Elitista

**Paso 1 — Preservar la élite:**

Se seleccionan los $e$ individuos con mayor aptitud de la generación actual:

$$\text{Elite}_g = \text{Top}_e(P_g) = \{I \in P_g : F(I) \text{ está entre los } e \text{ mejores}\}$$

**Paso 2 — Combinar con la nueva generación:**

$$P_{g+1}^{comb} = \text{Elite}_g \cup \text{Hijos}_g$$

**Paso 3 — Seleccionar los $N$ mejores para la nueva generación:**

$$P_{g+1} = \text{Top}_N(P_{g+1}^{comb})$$

### 8.3 Ejemplo Numérico

```
Generación g (N = 6 individuos):
  I₁: F = 0.9821  ← Elite (1°)
  I₂: F = 0.8534
  I₃: F = 0.9102  ← Elite (2°)
  I₄: F = 0.7215
  I₅: F = 0.9456
  I₆: F = 0.6891

Hijos generados (4 hijos por cruza + mutación):
  H₁: F = 0.9200
  H₂: F = 0.8100
  H₃: F = 0.9650
  H₄: F = 0.7800

Combinación (2 élites + 4 hijos = 6 candidatos):
  I₁: F = 0.9821  ← Sobrevive (1°)
  H₃: F = 0.9650  ← Sobrevive (2°)
  I₃: F = 0.9102  ← Sobrevive (3°)
  H₁: F = 0.9200  ← Sobrevive (4°)
  H₂: F = 0.8100  ← Sobrevive (5°)
  H₄: F = 0.7800  ← Sobrevive (6°)

Generación g+1 = {I₁, H₃, I₃, H₁, H₂, H₄}
Nota: El mejor individuo (I₁) sobrevivió intacto gracias al elitismo.
```

### 8.4 Parámetros Configurables

| Parámetro        | Símbolo | Rango | Valor Por Defecto |
| ---------------- | ------- | ----- | ----------------- |
| Número de élites | $e$     | 1 – 5 | 2                 |

---

## 9. Poda de la Población

### 9.1 Concepto

La poda es un mecanismo complementario al reemplazo que asegura que la población mantenga un tamaño constante $N$ y elimine individuos de baja calidad que consumen recursos computacionales sin aportar valor genético.

### 9.2 Mecanismo de Poda

Tras el reemplazo, si la población combinada excede el tamaño $N$, se ordenan todos los individuos por aptitud descendente y se eliminan los que sobran:

$$P_{g+1} = \text{Sort}(P_{g+1}^{comb}, F, \text{desc})[:N]$$

### 9.3 Umbral de Poda

Opcionalmente, se puede definir un umbral mínimo de aptitud $F_{min}$ debajo del cual un individuo es eliminado independientemente del tamaño de la población:

$$P_{g+1} = \{I \in P_{g+1}^{comb} : F(I) \geq F_{min}\}$$

Si la poda reduce la población por debajo de $N$, los espacios vacantes se rellenan con individuos generados aleatoriamente (inyección de diversidad).

### 9.4 Parámetros Configurables

| Parámetro                | Símbolo   | Rango     | Valor Por Defecto |
| ------------------------ | --------- | --------- | ----------------- |
| Tasa de poda             | $r_p$     | 0.0 – 0.3 | 0.1               |
| Umbral mínimo de aptitud | $F_{min}$ | 0.0 – 0.5 | 0.0 (desactivado) |

---

## 10. Condición de Término

### 10.1 Criterios de Terminación

El ciclo evolutivo se detiene cuando se cumple **al menos una** de las siguientes condiciones:

**Criterio 1 — Máximo de generaciones alcanzado:**

$$g \geq G_{max}$$

donde $G_{max}$ es el número máximo de generaciones permitido por fotograma.

**Criterio 2 — Umbral de error alcanzado:**

$$E(M_{mejor}, T) < \epsilon$$

donde $\epsilon$ es el umbral de error mínimo aceptable. Cuando se alcanza, la cara del robot ya es "suficientemente buena".

**Criterio 3 — Estancamiento evolutivo:**

$$|F_{max}^{(g)} - F_{max}^{(g-w)}| < \sigma$$

donde $w$ es la ventana de observación (número de generaciones hacia atrás) y $\sigma$ es el umbral de cambio mínimo. Si la mejor aptitud no mejora en $w$ generaciones, el AG se detuvo y no vale la pena seguir iterando.

### 10.2 Condición en Tiempo Real

En el contexto del sistema EVA, que opera fotograma por fotograma, la condición de término más importante es la **Condición 1** (máximo de generaciones). El sistema tiene un presupuesto finito de tiempo por fotograma (por ejemplo, 33 ms a 30 FPS) y debe entregar la mejor solución encontrada dentro de ese plazo:

$$t_{AG} \leq \frac{1}{\text{FPS}} = \frac{1}{30} \approx 33.3 \text{ ms}$$

### 10.3 Parámetros configurables

| Parámetro                              | Símbolo    | Rango          | Valor Por Defecto |
| -------------------------------------- | ---------- | -------------- | ----------------- |
| Máximo de generaciones (por fotograma) | $G_{max}$  | 5 – 100        | 20                |
| Umbral de error aceptable              | $\epsilon$ | 0.0001 – 0.01  | 0.001             |
| Ventana de estancamiento               | $w$        | 3 – 20         | 5                 |
| Umbral de estancamiento                | $\sigma$   | 0.0001 – 0.001 | 0.0001            |

---

## 11. Métricas de Estabilidad y Diversidad

### 11.1 Estabilidad entre Fotogramas ($S$)

La estabilidad mide cuánto cambia la solución del AG entre dos fotogramas consecutivos. Si las tensiones "saltan" bruscamente entre un fotograma y el siguiente, el avatar robótico mostrará espasmos visuales.

**Fórmula:**

$$S^{(f)} = \frac{1}{n} \sum_{i=1}^{n} \left( m_i^{(f)} - m_i^{(f-1)} \right)^2$$

donde:

- $m_i^{(f)}$ es la tensión del actuador $i$ del mejor individuo en el fotograma $f$
- $m_i^{(f-1)}$ es la tensión del mismo actuador en el fotograma anterior

**Interpretación:**

- $S \approx 0$: La cara del robot mantiene su expresión establemente (ideal durante una sonrisa sostenida)
- $S > 0.05$: Los actuadores están cambiando rápidamente (aceptable durante una transición de expresión)
- $S > 0.2$: Movimientos bruscos e indeseados (probable convergencia prematura o configuración incorrecta)

### 11.2 Diversidad Poblacional ($D$)

La diversidad mide la dispersión genética de la población. Se calcula como la distancia promedio entre todos los pares de individuos:

$$D^{(g)} = \frac{2}{N(N-1)} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} d_H(I_i, I_j)$$

donde $d_H(I_i, I_j)$ es la **distancia de Hamming** entre los cromosomas de los individuos $I_i$ e $I_j$:

$$d_H(I_i, I_j) = \sum_{l=1}^{L} \mathbb{1}[b_l^{(i)} \neq b_l^{(j)}]$$

Es decir, el número de posiciones de bits en las que difieren los dos cromosomas.

**Diversidad normalizada (entre 0 y 1):**

$$D_{norm}^{(g)} = \frac{D^{(g)}}{L}$$

**Interpretación:**

- $D_{norm} < 0.1$: Población homogénea (riesgo de convergencia prematura)
- $0.2 \leq D_{norm} \leq 0.4$: Rango saludable (equilibrio exploración/explotación)
- $D_{norm} > 0.5$: Población demasiado dispersa (el AG no está convergiendo)

---

## 12. Validación Cruzada (Cross-Validation)

### 12.1 Objetivo

La validación cruzada verifica que los hiperparámetros del AG ($p_c$, $p_m$, $k$, $N$, etc.) no producen resultados que solo funcionen bien para una cara o expresión específica (sobreajuste), sino que generalizan correctamente ante múltiples expresiones faciales.

### 12.2 Metodología K-Fold

Los datos recopilados durante una sesión experimental se segmentan en $K$ particiones temporales:

**Paso 1 — Recolección de datos:**
Durante una sesión de $T_{total}$ fotogramas, el sistema registra en SQLite para cada fotograma $f$:

- El vector objetivo $T^{(f)}$ (la cara del humano en ese instante)
- El vector solución $M^{(f)}$ (tensiones calculadas por el AG)
- El error $E^{(f)}$ y la aptitud $F^{(f)}$
- Los hiperparámetros activos

**Paso 2 — Partición de los datos:**

$$\text{Datos} = \{D_1, D_2, ..., D_K\}$$

donde cada $D_k$ contiene $\lfloor T_{total} / K \rfloor$ fotogramas consecutivos.

**Paso 3 — Evaluación por iteración:**

Para cada iteración $k = 1, ..., K$:

- **Conjunto de validación:** $D_k$
- **Conjunto de entrenamiento:** $\text{Datos} \setminus D_k$

Se ejecuta el AG con los hiperparámetros actuales sobre los datos de entrenamiento y se evalúa el error sobre el conjunto de validación:

$$E_{val}^{(k)} = \frac{1}{|D_k|} \sum_{f \in D_k} E(M^{(f)}, T^{(f)})$$

**Paso 4 — Error de generalización:**

$$E_{CV} = \frac{1}{K} \sum_{k=1}^{K} E_{val}^{(k)}$$

$$\sigma_{CV} = \sqrt{\frac{1}{K-1} \sum_{k=1}^{K} (E_{val}^{(k)} - E_{CV})^2}$$

### 12.3 Interpretación de resultados

| Resultado                         | Interpretación              | Acción                          |
| --------------------------------- | --------------------------- | ------------------------------- |
| $E_{CV}$ bajo, $\sigma_{CV}$ bajo | El AG generaliza bien       | Hiperparámetros óptimos         |
| $E_{CV}$ bajo, $\sigma_{CV}$ alto | Rendimiento inconsistente   | Aumentar diversidad o población |
| $E_{CV}$ alto, $\sigma_{CV}$ bajo | Error consistentemente alto | Ajustar $p_c$, $p_m$ o $N$      |
| $E_{CV}$ alto, $\sigma_{CV}$ alto | Sistema inestable y malo    | Rediseñar función de aptitud    |

### 12.4 Parámetros Configurables

| Parámetro                            | Símbolo   | Rango      | Valor Por Defecto  |
| ------------------------------------ | --------- | ---------- | ------------------ |
| Número de particiones                | $K$       | 3 – 10     | 5                  |
| Tamaño mínimo de sesión (fotogramas) | $T_{min}$ | 300 – 3000 | 900 (≈30s a 30FPS) |

---

## 13. Flujo Completo del Sistema

### 13.1 Diagrama de Flujo por Fotograma

```
╔══════════════════════════════════════════════════════════════════╗
║                    INICIO DEL FOTOGRAMA f                       ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
                ┌──────────────────────────┐
                │ 1. CAPTURA DE VIDEO       │
                │    Cámara → Fotograma RGB │
                │    (320×240, 15-30 FPS)   │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ 2. DETECCIÓN FACIAL       │
                │    MediaPipe Landmarker   │
                │    478 puntos + 52 blend  │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ 3. EXTRACCIÓN DEL TARGET  │
                │    T = [t₁, t₂,..., t₆]   │
                │    6 blendshapes → 6 vals │
                └──────────┬───────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │ 4. ¿PRIMER FOTOGRAMA?                     │
        │    SI → Generar Población Inicial P₀      │
        │    NO → Reutilizar población anterior P_g │
        └──────────────────┬───────────────────────┘
                           │
              ╔════════════╧════════════╗
              ║  CICLO EVOLUTIVO (AG)   ║
              ║  Repetir G_max veces    ║
              ╚════════════╤════════════╝
                           │
                           ▼
                ┌──────────────────────────┐
                │ 5. EVALUACIÓN             │
                │    ∀ Iⱼ ∈ P_g:            │
                │    Decodificar cromosoma  │
                │    Calcular E(Mⱼ, T)      │
                │    Calcular F(Mⱼ, T)      │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │ 6. ¿CONDICIÓN DE TÉRMINO? │
                │    E < ε  → TERMINAR      │
                │    Estancamiento → TERMIN. │
                └────┬─────────────┬───────┘
                     │ NO          │ SÍ
                     ▼             ▼
        ┌────────────────┐    (ir al paso 10)
        │ 7. SELECCIÓN    │
        │    Torneo k=3   │
        │    N/2 parejas   │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ 8. CRUZAMIENTO  │
        │    p_c = 0.8    │
        │    1 ó 2 puntos │
        │    → N hijos    │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ 9. MUTACIÓN     │
        │    p_m = 0.01   │
        │    Bit-flip     │
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────────────────┐
        │ 10. REEMPLAZO + ELITISMO   │
        │     Preservar e=2 mejores  │
        │     Combinar + Ordenar     │
        └────────┬───────────────────┘
                 │
                 ▼
        ┌────────────────────────────┐
        │ 11. PODA                    │
        │     Mantener N individuos   │
        │     Eliminar los peores     │
        └────────┬───────────────────┘
                 │
                 ▼
        (Volver al paso 5 si g < G_max)
                 │
                 ▼
┌────────────────────────────────────────────┐
│ 12. SALIDA DEL FOTOGRAMA                   │
│     M* = mejor cromosoma decodificado      │
│     → Renderizar en Pygame                  │
│     → Registrar en SQLite (E, F, M*, D, S) │
│     → Mostrar etiquetas en pantalla         │
└────────────────────────────────────────────┘
                 │
                 ▼
╔══════════════════════════════════════════════════════════════════╗
║                 FIN DEL FOTOGRAMA f → f+1                       ║
╚══════════════════════════════════════════════════════════════════╝
```

### 13.2 Resumen de Todos los Parámetros Configurables

| #   | Parámetro                         | Símbolo         | Valor Por Defecto |
| --- | --------------------------------- | --------------- | ----------------- |
| 1   | Tamaño de población               | $N$             | 50                |
| 2   | Bits por actuador                 | $k$             | 10                |
| 3   | Número de actuadores              | $n$             | 6                 |
| 4   | Longitud del cromosoma            | $L = n \cdot k$ | 60                |
| 5   | Tamaño del torneo                 | $k_t$           | 3                 |
| 6   | Tasa de selección                 | $r_s$           | 1.0               |
| 7   | Probabilidad de cruza             | $p_c$           | 0.8               |
| 8   | Tipo de cruza                     | —               | 1 punto           |
| 9   | Probabilidad de mutación          | $p_m$           | 0.01              |
| 10  | Número de élites                  | $e$             | 2                 |
| 11  | Tasa de poda                      | $r_p$           | 0.1               |
| 12  | Máximo generaciones por fotograma | $G_{max}$       | 20                |
| 13  | Umbral de error aceptable         | $\epsilon$      | 0.001             |
| 14  | Ventana de estancamiento          | $w$             | 5                 |
| 15  | Umbral de estancamiento           | $\sigma$        | 0.0001            |
| 16  | Folds de validación cruzada       | $K$             | 5                 |
| 17  | Resolución de la cámara           | —               | 320×240           |
| 18  | FPS de captura                    | —               | 15                |

---

## Referencias Bibliográficas

1. Chen, B., Hu, Y., Li, L., Cummings, S., & Lipson, H. (2021). _Smile Like You Mean It: Driving Animatronic Robotic Face with Learned Models._ arXiv:2105.12724.
2. Faraj, Z., Selamet, M., Morales, C., Torres, P., Hossain, M., Chen, B., & Lipson, H. (2020). _Facially expressive humanoid robotic face._ HardwareX, DOI: 10.1016/j.ohx.2020.e00117.
3. Holland, J. H. (1975). _Adaptation in Natural and Artificial Systems._ University of Michigan Press.
4. Goldberg, D. E. (1989). _Genetic Algorithms in Search, Optimization, and Machine Learning._ Addison-Wesley.
5. Ekman, P., & Friesen, W. V. (1978). _Facial Action Coding System._ Consulting Psychologists Press.
6. Lugaresi, C., et al. (2019). _MediaPipe: A Framework for Building Perception Pipelines._ arXiv:1906.08172.
