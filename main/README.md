## Requisitos Previos

- **Python 3.8 o superior**
- Una cámara web funcional.

---

## Guía de Instalación y Uso

### En Linux (Ubuntu/Debian)

1. **Crear entorno virtual:**
   ```bash
   python3 -m venv venv
   ```
2. **Activar el entorno:**
   ```bash
   source venv/bin/activate
   ```
3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ejecutar el programa:**
   ```bash
   python3 main.py
   ```

### En Windows

1. **Crear entorno virtual:**
   ```powershell
   python -m venv venv
   ```
2. **Activar el entorno:**
   ```powershell
   .\venv\Scripts\activate
   ```
3. **Instalar dependencias:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Ejecutar el programa:**
   ```powershell
   python main.py
   ```

---

## Controles e Interfaz

- **Teclas:**
  - `q`: Cerrar el sistema de forma segura y generar reportes.
- **Interacción con el Ratón:**
  - Haz **clic izquierdo** en el botón **"Mapeo de Motores"** (esquina superior derecha) para alternar la visualización técnica de los actuadores y tensores.

---

## Salidas del Sistema

1. **Gemelo Digital:** Visualización en tiempo real del rostro azul rey reaccionando a tus gestos.
2. **Base de Datos:** Los datos se guardan automáticamente en `data/eva_conocimiento.db`.
3. **Reportes:** Al cerrar el programa con `q`, se generarán automáticamente gráficas de rendimiento en la carpeta `output/reportes/` (dentro de este mismo directorio).

---

## Dependencias

Todas las librerías necesarias están en `requirements.txt`:

| Librería        | Uso                                         |
| --------------- | ------------------------------------------- |
| `opencv-python` | Captura de video y procesamiento de imagen  |
| `mediapipe`     | Detección de landmarks faciales             |
| `pygame`        | Renderizado del gemelo digital 2D           |
| `matplotlib`    | Generación de gráficas de reporte           |
| `numpy`         | Operaciones vectoriales y cálculo matricial |
| `pandas`        | Manejo de datos para análisis estadístico   |

---

## Estructura de Carpetas

- `main.py`: Lanzador principal.
- `data/`: Contiene la base de datos SQLite con los hiperparámetros y el historial.
- `client/`: Módulos de interfaz (MediaPipe, Pygame).
- `core/`: Núcleo del Algoritmo Genético e infraestructura de datos.
- `output/reportes/`: Carpeta de salida para las gráficas generadas al final de cada sesión.
- `requirements.txt`: Lista de librerías necesarias.
