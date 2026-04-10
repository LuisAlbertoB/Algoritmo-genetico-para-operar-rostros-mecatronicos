"""
capturaVideo.py
===============
§8 Capa 1 — Fuente de Video Crudo.

Gestiona la cámara web mediante OpenCV, proporcionando
un flujo continuo de fotogramas al sistema de detección facial.

Parámetros configurables (según propuesta académica §8.1):
    - Resolución espacial: 320×240 por defecto
    - FPS: 10-30 según capacidad del equipo
    - Índice de dispositivo: 0 (cámara integrada)
"""

import cv2


class CapturaVideo:
    """
    Gestiona la captura de video desde la cámara web.

    Uso:
        with CapturaVideo() as cam:
            ok, frame = cam.leer_frame()
    """

    def __init__(self, indice_camara: int = 0,
                 ancho: int = 320, alto: int = 240, fps: int = 30):
        """
        Inicializa la cámara.

        Args:
            indice_camara: Índice del dispositivo (0 = cámara integrada).
            ancho:         Resolución horizontal en píxeles.
            alto:          Resolución vertical en píxeles.
            fps:           Fotogramas por segundo deseados.
        """
        self.indice = indice_camara
        self.ancho = ancho
        self.alto = alto
        self.fps = fps
        self.captura = None

    def iniciar(self) -> bool:
        """
        Abre la cámara y aplica la configuración de resolución y FPS.

        Returns:
            True si la cámara se abrió correctamente, False en caso contrario.
        """
        self.captura = cv2.VideoCapture(self.indice)

        if not self.captura.isOpened():
            print(f"  ⚠ Error: No se pudo abrir la cámara {self.indice}")
            return False

        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
        self.captura.set(cv2.CAP_PROP_FPS, self.fps)

        # Leer resolución real (puede diferir de la solicitada)
        w_real = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_real = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_real = self.captura.get(cv2.CAP_PROP_FPS)

        print(f"  📷 Cámara {self.indice} abierta: {w_real}×{h_real} @ {fps_real:.0f} FPS")
        return True

    def leer_frame(self):
        """
        Lee el siguiente fotograma de la cámara.

        Returns:
            Tupla (éxito: bool, frame: numpy.ndarray o None).
        """
        if self.captura is None or not self.captura.isOpened():
            return False, None

        return self.captura.read()

    def liberar(self):
        """Libera los recursos de la cámara."""
        if self.captura is not None and self.captura.isOpened():
            self.captura.release()
            print("  📷 Cámara liberada.")

    # ── Context Manager ─────────────────────────────────────────
    def __enter__(self):
        self.iniciar()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.liberar()
        return False
