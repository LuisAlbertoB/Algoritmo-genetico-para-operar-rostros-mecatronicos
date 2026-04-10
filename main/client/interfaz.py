"""
interfaz.py
===========
§8 Capa 3 — Orquestador de la Interfaz de Entrada.

Une la cámara (Capa 1), el detector facial (Capa 2) y la función
de mapeo extraccionTarget (Capa 3) en un módulo cohesivo que
produce el vector T = [t₁,...,t₆] para el AG.

Renderiza una ventana OpenCV con:
    - Video en vivo de la cámara
    - Malla de landmarks superpuesta
    - Valores de blendshapes relevantes
    - Vector T actual en pantalla
"""

import cv2
import sys
import os
import time
from typing import List, Optional

# Imports del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core', 'AG'))
from capturaVideo import CapturaVideo
from deteccionFacial import DetectorFacial
from extraccionTarget import construir_vector_target, BLENDSHAPES_REQUERIDOS


class InterfazEntrada:
    """
    Orquesta la captura de video, detección facial y construcción
    del vector target T para alimentar al AG.

    Atributos:
        vector_T: Vector objetivo actual [t₁,...,t₆].
        rostro_detectado: Si se detectó un rostro en el último frame.
        fps_actual: FPS medidos en tiempo real.
    """

    NOMBRE_VENTANA = "EVA - Entrada (Camara + Tracking)"

    NOMBRES_ACTUADORES = [
        'm1(Boca)', 'm2(CigI)', 'm3(CigD)',
        'm4(OjI)',  'm5(OjD)',  'm6(Cejas)'
    ]

    def __init__(self, indice_camara: int = 0,
                 ancho: int = 320, alto: int = 240, fps: int = 30,
                 confianza: float = 0.5):
        """
        Args:
            indice_camara: Índice del dispositivo de cámara.
            ancho, alto:   Resolución de captura.
            fps:           Fotogramas por segundo deseados.
            confianza:     Umbral de confianza para MediaPipe.
        """
        self.camara = CapturaVideo(indice_camara, ancho, alto, fps)
        self.detector = DetectorFacial(confianza_min=confianza)

        self.vector_T: List[float] = [0.0] * 6
        self.rostro_detectado: bool = False
        self.fps_actual: float = 0.0

        self._tiempo_anterior = time.time()
        self._activa = False

    def iniciar(self) -> bool:
        """
        Inicializa cámara y detector facial.

        Returns:
            True si ambos componentes se iniciaron correctamente.
        """
        print("\n  ═══ INTERFAZ DE ENTRADA EVA ═══")

        if not self.camara.iniciar():
            return False

        if not self.detector.iniciar():
            self.camara.liberar()
            return False

        self._activa = True
        return True

    def procesar_frame(self):
        """
        Procesa un fotograma completo:
            1. Lee frame de la cámara
            2. Detecta rostro y extrae blendshapes
            3. Construye vector T
            4. Dibuja visualización

        Returns:
            Tupla (frame_renderizado, vector_T, rostro_detectado)
            o (None, vector_T, False) si no hay frame.
        """
        ok, frame = self.camara.leer_frame()
        if not ok or frame is None:
            return None, self.vector_T, False

        # ── Capa 2: Detección facial ────────────────────────────
        blendshapes, landmarks, detectado = self.detector.procesar(frame)
        self.rostro_detectado = detectado

        # ── Capa 3: Construir vector T ──────────────────────────
        if blendshapes:
            self.vector_T = construir_vector_target(blendshapes)

        # ── Renderizado visual ──────────────────────────────────
        frame_viz = self._renderizar(frame, landmarks, blendshapes, detectado)

        # ── FPS ─────────────────────────────────────────────────
        ahora = time.time()
        dt = ahora - self._tiempo_anterior
        self.fps_actual = 1.0 / dt if dt > 0 else 0.0
        self._tiempo_anterior = ahora

        return frame_viz, self.vector_T, detectado

    def obtener_target(self) -> List[float]:
        """Devuelve el vector T actual."""
        return list(self.vector_T)

    def mostrar(self, frame) -> bool:
        """
        Muestra el frame en la ventana OpenCV.

        Returns:
            False si el usuario presionó 'q' para salir.
        """
        if frame is not None:
            cv2.imshow(self.NOMBRE_VENTANA, frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            return False

        return True

    def liberar(self):
        """Libera todos los recursos."""
        self._activa = False
        self.detector.liberar()
        self.camara.liberar()
        cv2.destroyAllWindows()
        print("  ✓ Interfaz de entrada finalizada.")

    # ── Renderizado interno ─────────────────────────────────────

    def _renderizar(self, frame, landmarks, blendshapes, detectado):
        """Compone la visualización final del frame de entrada."""
        # Escalar para mejor visibilidad
        frame_viz = cv2.resize(frame, (640, 480))
        escala_x = 640 / frame.shape[1]
        escala_y = 480 / frame.shape[0]

        # Dibujar landmarks
        if landmarks:
            h, w, _ = frame_viz.shape
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame_viz, (x, y), 1, (0, 255, 0), -1)

        # Dibujar blendshapes relevantes (panel izquierdo)
        frame_viz = self.detector.dibujar_blendshapes_texto(
            frame_viz, blendshapes, BLENDSHAPES_REQUERIDOS
        )

        # Dibujar vector T (panel inferior)
        self._dibujar_vector_T(frame_viz)

        # Indicador de detección
        estado = "ROSTRO DETECTADO" if detectado else "SIN DETECCION"
        color = (0, 255, 0) if detectado else (0, 0, 255)
        cv2.putText(frame_viz, estado, (640 - 200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # FPS
        cv2.putText(frame_viz, f"FPS: {self.fps_actual:.1f}", (640 - 100, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame_viz

    def _dibujar_vector_T(self, frame):
        """Dibuja el vector T actual como barras horizontales en la parte inferior."""
        h, w, _ = frame.shape
        y_base = h - 120

        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_base - 5), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "Vector T (Target para AG):", (5, y_base + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        for i, (nombre, valor) in enumerate(zip(self.NOMBRES_ACTUADORES, self.vector_T)):
            y = y_base + 25 + i * 16

            # Etiqueta
            cv2.putText(frame, f"{nombre}:", (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Barra de progreso
            barra_ancho = int(valor * 200)
            cv2.rectangle(frame, (100, y - 10), (100 + 200, y - 2), (50, 50, 50), -1)
            color_barra = (0, int(255 * (1 - valor)), int(255 * valor))
            cv2.rectangle(frame, (100, y - 10), (100 + barra_ancho, y - 2), color_barra, -1)

            # Valor numérico
            cv2.putText(frame, f"{valor:.3f}", (310, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


# ═══════════════════════════════════════════════════════════════════
#  EJECUCIÓN STANDALONE (prueba independiente de entrada)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    interfaz = InterfazEntrada()

    if not interfaz.iniciar():
        print("  ⚠ No se pudo iniciar la interfaz de entrada.")
        sys.exit(1)

    print("  Presiona 'q' para salir.\n")

    try:
        while True:
            frame, T, detectado = interfaz.procesar_frame()
            if not interfaz.mostrar(frame):
                break
    except KeyboardInterrupt:
        pass
    finally:
        interfaz.liberar()
