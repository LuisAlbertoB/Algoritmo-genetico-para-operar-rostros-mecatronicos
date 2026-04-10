# Propuesta de Proyecto Académico

## Título

**Sistema inteligente para calcular las tensiones mecánicas de un rostro robótico mediante algoritmos genéticos para la imitación de expresiones faciales humanas, basado en el caso EVA**

---

## 1. Contexto de la Problemática

Actualmente, los robots trabajan cada vez más cerca de las personas en lugares como asilos, hospitales, fábricas y hogares. Sin embargo, casi todos estos robots tienen rostros estáticos hechos de plástico o metal rígido, incapaces de mostrar emociones.

Esto genera una importante barrera de **confianza humano-robot**: a las personas les resulta difícil conectar o sentirse cómodas colaborando con máquinas que tienen rostros inexpresivos (estilo "jugador de póquer"). Para solucionar este problema acústico y social, proyectos como el robot EVA proponen crear rostros robóticos de silicona suave, controlados internamente por decenas de pequeños motores y cables que actúan como "músculos artificiales".

El gran **problema técnico** surge al intentar controlar esta nueva tecnología física. Coordinar mecánicamente la tensión exacta de más de 40 pequeños motores para recrear una sonrisa o un gesto humano natural es tan complejo que resulta prácticamente imposible de programar a mano usando reglas matemáticas tradicionales. La elasticidad de la silicona, el desgaste de las piezas y la fricción hacen que los movimientos del rostro sean altamente impredecibles.

---

## 2. Enunciado de la Problemática

La incapacidad de calcular y coordinar con precisión las tensiones mecánicas de los múltiples músculos artificiales provoca que el rostro de silicona del robot genere gesticulaciones espasmódicas, erráticas o gravemente antinaturales. Este fallo motriz desencadena un fuerte rechazo psicológico en los usuarios humanos (fenómeno conocido como el "valle inquietante"), lo que destruye cualquier posibilidad de empatía o confianza humano-máquina y, en última instancia, provoca el fracaso comercial y operativo del robot en entornos donde su labor asistencial, médica o colaborativa es crítica.

---

## 3. Descripción del Proyecto

El proyecto consiste en el desarrollo de un sistema de simulación basado en algoritmos genéticos que toma como entrada las coordenadas faciales en tiempo real de un usuario capturado por video, procesándolas para calcular la tensión motriz óptima de los múltiples cables y actuadores de un rostro robótico blando (basado en la arquitectura EVA), buscando replicar fielmente la expresión humana en un gemelo digital 2D para superar las limitaciones de la cinemática inversa tradicional.

---

## 4. Variables de Decisión

Las variables de decisión representan la tensión mecánica aplicada a cada actuador (motor/cable) del rostro robótico, expresada en valores continuos dentro del rango $[0.0, 1.0]$, donde $0.0$ representa tensión nula (músculo completamente relajado) y $1.0$ representa tensión máxima (músculo completamente contraído).

- **$m_1$ — Tensión del actuador mandibular (apertura de boca):** Controla el desplazamiento vertical de la mandíbula inferior del rostro robótico.
- **$m_2$ — Tensión del actuador cigomático izquierdo (comisura labial izquierda):** Controla la elevación o descenso del extremo izquierdo de la boca, determinando la curvatura de la sonrisa en ese lado.
- **$m_3$ — Tensión del actuador cigomático derecho (comisura labial derecha):** Equivalente al anterior para el lado derecho del rostro.
- **$m_4$ — Tensión del actuador orbicular izquierdo (párpado izquierdo):** Controla el grado de apertura o cierre del ojo izquierdo.
- **$m_5$ — Tensión del actuador orbicular derecho (párpado derecho):** Controla el grado de apertura o cierre del ojo derecho.
- **$m_6$ — Tensión del actuador frontal (músculos de las cejas):** Controla la elevación o el fruncimiento de las cejas, expresando estados como sorpresa, enojo o concentración.

---

## 5. Variables a Optimizar

Las variables a optimizar son los resultados observables del sistema que se buscan mejorar en función de los valores que tomen las variables de decisión ($m_1$ ... $m_6$). El Algoritmo Genético buscará minimizar o maximizar cada una de ellas en cada generación.

- **$E$ — Error de imitación facial (MSE):** Es la variable principal a **minimizar**. Representa la distancia cuadrática media entre el vector de coordenadas faciales del usuario humano capturado por la cámara y el vector de posición resultante del gemelo digital tras aplicar las tensiones calculadas. Un valor de $E = 0$ significaría una imitación perfecta de la expresión.

- **$F$ — Aptitud de convergencia (Fitness):** Es la variable principal a **maximizar**. Se calcula como $F = \frac{1}{1 + E}$, de modo que cuanto menor sea el error $E$, mayor será la aptitud $F$. Es la función que el Algoritmo Genético evalúa en cada individuo para decidir su supervivencia.

- **$V$ — Velocidad de convergencia generacional:** Representa el número de generaciones que el AG requiere para alcanzar un umbral de error aceptable $E < \epsilon$ ante una expresión facial nueva. Se busca **minimizar** este valor para garantizar una respuesta fluida y en tiempo real del sistema.

- **$S$ — Estabilidad de la solución entre fotogramas:** Mide la variación promedio del vector $[m_1, ..., m_6]$ entre fotogramas consecutivos de video. Una alta inestabilidad genera efectos visuales espasmódicos en el gemelo digital. Se busca **minimizar** este valor para garantizar transiciones faciales suaves y naturales.

- **$D$ — Diversidad de la población genética:** Mide la dispersión de los cromosomas activos dentro de la población del AG. Se busca mantener este valor en un rango óptimo: si es demasiado baja, el AG converge prematuramente en un mínimo local (expresión incorrecta pero estable); si es demasiado alta, el AG no converge y la cara del robot "tiembla" sin control.

---

## 6. Objetivos de Optimización

Cada variable a optimizar tiene un objetivo explícito que el Algoritmo Genético perseguirá durante cada ciclo evolutivo:

- **Minimizar el error de imitación facial ($E$):** Reducir al mínimo posible la diferencia cuadrática media entre las coordenadas faciales del usuario humano y las coordenadas generadas por el gemelo digital del robot, logrando la mayor fidelidad posible en la réplica de la expresión.

- **Maximizar la aptitud de convergencia ($F$):** Incrementar el valor de la función de aptitud $F = \frac{1}{1+E}$ de la mejor solución encontrada en cada generación, asegurando que el AG evolucione consistentemente hacia configuraciones de tensión motriz más precisas.

- **Minimizar la velocidad de convergencia generacional ($V$):** Reducir el número de generaciones requeridas por el AG para alcanzar un umbral de error aceptable, garantizando que el sistema responda de forma fluida y en tiempo real ante cambios en la expresión del usuario.

- **Minimizar la inestabilidad de la solución entre fotogramas ($S$):** Reducir la variación brusca del vector de tensiones $[m_1, ..., m_6]$ entre fotogramas consecutivos, eliminando los movimientos espasmódicos del rostro robótico virtual y asegurando transiciones faciales suaves y naturales.

- **Equilibrar la diversidad de la población genética ($D$):** Mantener la diversidad cromosómica dentro de un rango óptimo que evite tanto la convergencia prematura hacia mínimos locales (diversidad baja) como la falta total de convergencia que impediría al robot adoptar una expresión estable (diversidad alta).

---

## 7. Base de Conocimiento

La Base de Conocimiento contiene toda la información estática, propia del modelo robótico simulado, que el sistema utiliza como referencia para calcular la aptitud de cada individuo. Esta información es **preexistente, no modificable** durante la ejecución del AG y se almacena en la tabla estática de la base de datos SQLite del sistema.

- **Modelo del Gemelo Digital (Descripción del Robot Simulado):** Catálogo del rostro robótico virtual que define cuántos actuadores lo componen ($n = 6$), el nombre y rol biomecánico de cada actuador ($m_1$...$m_6$), y los límites máximos y mínimos de tensión de cada uno (rango $[0.0, 1.0]$).

- **Tabla de Mapeo Cinemático 2D:** Conjunto de reglas que describen cómo cada combinación de tensiones de los actuadores desplaza visualmente los puntos clave del rostro en el gemelo digital (Pygame). Por ejemplo: si $m_2 = 1.0$ y $m_3 = 1.0$, ambas comisuras labiales ascienden y el sistema renderiza una sonrisa completa.

- **Tabla de Expresiones de Referencia (6 Emociones Básicas):** Catálogo de los 6 vectores de tensión ideales $[m_1,...,m_6]$ correspondientes a las emociones básicas universales (alegría, tristeza, sorpresa, miedo, enojo y asco), según la taxonomía de Paul Ekman. Se utilizan como referencia de calibración y para la validación cruzada.

- **Parámetros Biomecánicos del Material:** Constantes que modelan las propiedades físicas del material de silicona simulado, como el coeficiente de elasticidad y los límites de deformación, que escalan la señal de los actuadores hacia el movimiento real del píxel en pantalla.

- **Configuración Inicial del AG (Parámetros del Sistema):** Valores por defecto del sistema genético, como tamaño de población, tasa de mutación, tasa de cruza, número de generaciones y umbral de error aceptable ($\epsilon$). Estos son definidos por el usuario antes de la ejecución y sirven como punto de partida controlable del experimento.

### 7.1 Descripción de la Base de Datos Estática (Componente SQLite de la Base de Conocimiento)

La Base de Conocimiento se implementa físicamente como un conjunto de **tablas estáticas dentro de un archivo SQLite** (`eva_conocimiento.db`). A diferencia de las tablas dinámicas que registran los resultados del AG en tiempo de ejecución, estas tablas son pobladas **una única vez antes de iniciar el sistema** y permanecen en modo solo-lectura durante toda la duración del experimento.

Esta separación garantiza que el Algoritmo Genético nunca altere las reglas del problema que intenta resolver: es equivalente a asegurar que un estudiante resuelva un examen sin poder modificar las preguntas.

La base de datos estática se organiza en cinco tablas:

| Tabla | Contenido | Uso por el AG |
|---|---|---|
| `robot_modelo` | Nombre, cantidad y rango de cada actuador | Define el tamaño del cromosoma (bits por gen) |
| `mapeo_cinematico` | Reglas de transformación tensión → pixel | Convierte la solución del AG en imagen visual |
| `expresiones_ekman` | 6 vectores de expresión de referencia | Calibración y validación cruzada |
| `parametros_material` | Coeficientes de elasticidad y deformación | Escala el movimiento en el gemelo digital |
| `config_ag` | Hiperparámetros iniciales del sistema genético | Punto de partida controlable por el usuario |

Cuando el AG necesita calcular la aptitud de un individuo, consulta en tiempo real la tabla `mapeo_cinematico` para saber qué expresión producen las tensiones del cromosoma actual y la compara contra el vector facial capturado por la cámara. Todos los cálculos de penalización o bonificación parten de estas tablas, nunca de valores codificados directamente en el código fuente. Esto garantiza que cambiar el modelo del robot (por ejemplo, pasar de 6 a 12 actuadores) sea tan simple como modificar una fila en la tabla `robot_modelo`, sin tocar una sola línea del algoritmo genético.

---

## 8. Entradas al Sistema

Las entradas al sistema son la información que el sistema recibe para realizar la optimización. Aunque la fuente principal es la cámara de video, ésta por sí sola no constituye una entrada válida para el AG: la imagen cruda debe pasar por una **capa de procesamiento visual** que la transforma en datos numéricos estructurados. El sistema reconoce tres capas de entrada:

### Capa 1 — Fuente de Video Crudo (Cámara Web)
El dispositivo de captura proporciona el flujo continuo de fotogramas. Sus parámetros son configurables por el usuario:

- **Resolución espacial:** Reducida a 320×240 píxeles para optimizar el rendimiento en hardware limitado.
- **Frecuencia de muestreo (FPS):** Configurable entre 10 y 30 FPS según la capacidad del equipo.
- **Índice del dispositivo:** Identificador numérico de la cámara activa (por defecto: `0`, cámara integrada).

### Capa 2 — Vector de Puntos Faciales (Salida de MediaPipe)
Cada fotograma es procesado por MediaPipe FaceLandmarker, transformando la imagen cruda en datos numéricos que el AG puede procesar:

- **Coordenadas de puntos clave (Landmarks):** 478 puntos faciales normalizados en $[0.0, 1.0]$, expresados como pares $(x, y)$.
- **Blendshapes faciales:** 52 valores continuos que cuantifican rasgos específicos: `jawOpen`, `mouthSmileLeft`, `mouthSmileRight`, `eyeBlinkLeft`, `eyeBlinkRight`, `browDownLeft`, `browDownRight`.
- **Nivel de confianza de detección:** Filtro de calidad que descarta fotogramas con detección insuficiente, reutilizando el último vector válido.

### Capa 3 — Vector Objetivo del AG (Target $T$)
A partir de los blendshapes de la Capa 2, el sistema construye el **vector objetivo** $T = [t_1, t_2, t_3, t_4, t_5, t_6]$, entrada real y directa al Algoritmo Genético:

| Componente | Blendshape de origen | Actuador correspondiente |
|---|---|---|
| $t_1$ | `jawOpen` | $m_1$ — Mandibular (apertura de boca) |
| $t_2$ | `mouthSmileLeft` | $m_2$ — Cigomático izquierdo |
| $t_3$ | `mouthSmileRight` | $m_3$ — Cigomático derecho |
| $t_4$ | `1.0 - eyeBlinkLeft` | $m_4$ — Orbicular izquierdo |
| $t_5$ | `1.0 - eyeBlinkRight` | $m_5$ — Orbicular derecho |
| $t_6$ | `1.0 - browDownLeft` | $m_6$ — Frontal (cejas) |

Este vector $T$ es el único dato que el AG consume para evaluar su población. La función de aptitud calculará qué tan cerca están las tensiones del mejor cromosoma de reproducir fielmente los valores de $T$.

---

## 9. Salidas Esperadas del Sistema

El sistema genera dos tipos de salidas claramente diferenciadas según su momento de producción y su propósito.

### 9.1 Salidas en Tiempo Real
Se producen continuamente durante la ejecución y están diseñadas para no comprometer el rendimiento del AG.

- **Gemelo Digital Animado (Pygame — Principal):** Ventana de 480×480 píxeles que renderiza el rostro robótico virtual. Los movimientos del avatar reflejan en tiempo real las tensiones del **mejor cromosoma** de la generación actual, permitiendo observar visualmente cómo el AG converge hacia la expresión del usuario fotograma a fotograma.

- **Panel de Etiquetas en Pantalla (Secundario):** Superpuesto sobre el gemelo digital, muestra en tiempo real:
  - Generación actual y número de fotograma
  - Mejor aptitud ($F$) y error ($E$ / MSE) del mejor individuo
  - Valores actuales del vector $[m_1,...,m_6]$ del mejor individuo

- **Backtracking Visual de la Cámara (Terciario — Activable):** Al presionar una tecla, activa una segunda ventana con el fotograma original y la malla de landmarks de MediaPipe superpuesta, para verificar visualmente la calidad de la detección facial.

### 9.2 Salidas Estáticas (Bajo Demanda)
Se generan al finalizar la sesión consultando la base de datos SQLite de resultados.

- **Gráfica de Evolución de Aptitud:** Curvas de mejor aptitud ($F_{max}$) y aptitud promedio ($F_{prom}$) por generación, evidenciando la trayectoria de convergencia del AG.

- **Gráfica de Decaimiento del Error (MSE):** Curva del Error Cuadrático Medio ($E$) por generación, demostrando la reducción progresiva de la diferencia entre la expresión del usuario y la del robot simulado.

- **Gráfica de Diversidad Poblacional:** Evolución del índice $D$ por generación, identificando convergencia prematura o exploración saludable del espacio de soluciones.

- **Tabla de los Tres Mejores Individuos (Por Sesión):** Podio de los tres cromosomas con mayor aptitud registrados durante la sesión completa:

  | Posición | Cromosoma (bits) | $m_1$ | $m_2$ | $m_3$ | $m_4$ | $m_5$ | $m_6$ | Aptitud $F$ | Error $E$ | Generación |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 🥇 1° | `101011...` | 0.85 | 0.90 | 0.88 | 0.75 | 0.78 | 0.60 | 0.9821 | 0.0179 | 34 |
  | 🥈 2° | `100111...` | 0.82 | 0.87 | 0.85 | 0.72 | 0.74 | 0.58 | 0.9743 | 0.0257 | 31 |
  | 🥉 3° | `101001...` | 0.80 | 0.89 | 0.83 | 0.70 | 0.76 | 0.55 | 0.9695 | 0.0305 | 28 |

- **Reporte de Validación Cruzada:** Análisis estadístico $K$-fold sobre los datos de la sesión, reportando el error de generalización del sistema ante diferentes expresiones faciales y validando que el AG no sobreajustó su solución a un único gesto.