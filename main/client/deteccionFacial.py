"""
deteccionFacial.py
==================
§8 Capa 2 — Vector de Puntos Faciales (Salida de MediaPipe).

Utiliza MediaPipe FaceLandmarker para extraer:
    - 478 landmarks faciales normalizados
    - 52 blendshapes faciales (jawOpen, mouthSmileLeft, etc.)

Incluye filtro de confianza: si la detección es insuficiente,
reutiliza el último vector válido (§8 Capa 2).
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from typing import Dict, List, Optional, Tuple
import os
import urllib.request


# Ruta al modelo de MediaPipe (se descarga automáticamente si no existe)
MODELO_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
MODELO_PATH = os.path.join(MODELO_DIR, 'face_landmarker.task')
MODELO_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'


def descargar_modelo():
    """Descarga el modelo de MediaPipe si no existe localmente."""
    if os.path.exists(MODELO_PATH):
        return True

    os.makedirs(MODELO_DIR, exist_ok=True)
    print(f"  🔽 Descargando modelo FaceLandmarker...")
    try:
        urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
        print(f"  ✓ Modelo descargado en: {MODELO_PATH}")
        return True
    except Exception as e:
        print(f"  ⚠ Error al descargar modelo: {e}")
        return False


class DetectorFacial:
    """
    Envuelve MediaPipe FaceLandmarker para extraer blendshapes y landmarks.

    Atributos:
        ultimo_blendshapes: Último diccionario de blendshapes válido.
        ultimo_landmarks:   Últimos landmarks válidos.
        confianza_min:      Umbral mínimo de confianza para aceptar detección.
    """

    def __init__(self, confianza_min: float = 0.5):
        """
        Args:
            confianza_min: Confianza mínima de detección facial [0.0, 1.0].
        """
        self.confianza_min = confianza_min
        self.detector = None
        self.ultimo_blendshapes: Dict[str, float] = {}
        self.ultimo_landmarks: list = []
        self._rostro_detectado = False

    def iniciar(self) -> bool:
        """
        Inicializa el detector de MediaPipe.

        Returns:
            True si el modelo se cargó correctamente.
        """
        if not descargar_modelo():
            return False

        opciones_base = mp_python.BaseOptions(
            model_asset_path=MODELO_PATH
        )

        opciones = vision.FaceLandmarkerOptions(
            base_options=opciones_base,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=self.confianza_min,
            min_face_presence_confidence=self.confianza_min,
            min_tracking_confidence=self.confianza_min,
        )

        self.detector = vision.FaceLandmarker.create_from_options(opciones)
        print(f"  🧠 MediaPipe FaceLandmarker inicializado (confianza ≥ {self.confianza_min})")
        return True

    def procesar(self, frame: np.ndarray) -> Tuple[Dict[str, float], list, bool]:
        """
        §8 Capa 2 — Procesa un fotograma y extrae blendshapes + landmarks.

        Si la detección tiene confianza insuficiente, reutiliza los
        últimos valores válidos (según especificación §8 Capa 2).

        Args:
            frame: Imagen BGR de OpenCV (numpy array).

        Returns:
            Tupla (blendshapes_dict, landmarks_list, rostro_detectado).
        """
        if self.detector is None:
            return self.ultimo_blendshapes, self.ultimo_landmarks, False

        # Convertir BGR (OpenCV) → RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        resultado = self.detector.detect(imagen_mp)

        # ── Verificar que se detectó al menos un rostro ─────────
        if (resultado.face_landmarks and len(resultado.face_landmarks) > 0 and
                resultado.face_blendshapes and len(resultado.face_blendshapes) > 0):

            # Extraer landmarks del primer rostro
            self.ultimo_landmarks = resultado.face_landmarks[0]

            # Extraer blendshapes como diccionario {nombre: valor}
            self.ultimo_blendshapes = {
                bs.category_name: bs.score
                for bs in resultado.face_blendshapes[0]
            }
            self._rostro_detectado = True
        else:
            # Sin detección → mantener últimos válidos
            self._rostro_detectado = False

        return self.ultimo_blendshapes, self.ultimo_landmarks, self._rostro_detectado

    def dibujar_landmarks(self, frame: np.ndarray, landmarks: list,
                           color: tuple = (0, 255, 0), radio: int = 1) -> np.ndarray:
        """
        §9.1 Backtracking Visual — Dibuja los landmarks faciales sobre el frame.

        Args:
            frame:     Imagen BGR de OpenCV.
            landmarks: Lista de landmarks de MediaPipe.
            color:     Color BGR para los puntos.
            radio:     Radio de los círculos.

        Returns:
            Frame con landmarks dibujados.
        """
        if not landmarks:
            return frame

        frame_out = frame.copy()
        h, w, _ = frame_out.shape

        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame_out, (x, y), radio, color, -1)

        return frame_out

    def dibujar_blendshapes_texto(self, frame: np.ndarray,
                                   blendshapes: Dict[str, float],
                                   claves: list = None) -> np.ndarray:
        """
        Superpone los valores numéricos de los blendshapes relevantes
        sobre el frame para diagnóstico visual.

        Args:
            frame:       Imagen BGR de OpenCV.
            blendshapes: Diccionario de blendshapes.
            claves:      Lista de claves a mostrar (None = las 6 del AG).
        """
        if claves is None:
            claves = [
                'jawOpen', 'mouthSmileLeft', 'mouthSmileRight',
                'eyeBlinkLeft', 'eyeBlinkRight', 'browDownLeft'
            ]

        frame_out = frame.copy()
        y_offset = 20

        # Fondo semitransparente para legibilidad
        overlay = frame_out.copy()
        cv2.rectangle(overlay, (0, 0), (220, 20 + len(claves) * 18), (0, 0, 0), -1)
        frame_out = cv2.addWeighted(overlay, 0.6, frame_out, 0.4, 0)

        for clave in claves:
            valor = blendshapes.get(clave, 0.0)
            texto = f"{clave}: {valor:.3f}"
            cv2.putText(frame_out, texto, (5, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 18

        return frame_out

    def liberar(self):
        """Libera los recursos del detector."""
        if self.detector is not None:
            self.detector.close()
            self.detector = None
            print("  🧠 Detector MediaPipe liberado.")
