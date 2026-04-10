"""
salidaDelFotograma.py
=====================
§9.1 — Salidas en Tiempo Real del Sistema EVA.

Renderiza el gemelo digital 2D del rostro robótico usando Pygame.
Los movimientos del avatar reflejan las tensiones [m₁,...,m₆]
del mejor cromosoma de la generación actual.

Componentes visuales (§9.1):
    - Gemelo Digital Animado (ventana 480×480)
    - Panel de Etiquetas: generación, F, E, vector [m₁,...,m₆]
"""

import pygame
import sys
import math
import random
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class EstadoAG:
    """Datos del AG para mostrar en el panel de etiquetas."""
    generacion: int = 0
    fotograma: int = 0
    mejor_aptitud: float = 0.0
    mejor_error: float = float('inf')
    tensiones: List[float] = None
    razon_termino: str = ""

    def __post_init__(self):
        if self.tensiones is None:
            self.tensiones = [0.0] * 6


# ─── Colores del gemelo digital ─────────────────────────────────
FONDO          = (30,  30,  40)
PIEL           = (65,  105, 225)  # Azul Rey (Royal Blue)
PIEL_OSCURA    = (0,   35,  102)  # Azul Rey Oscuro (para bordes)
BLANCO_OJO     = (240, 240, 245)
IRIS           = (70,  130, 90)
PUPILA         = (20,  20,  25)
LABIO          = (180, 80,  80)
CEJA           = (80,  60,  50)
PANEL_BG       = (20,  20,  30)
TEXTO          = (200, 200, 210)
TEXTO_DESTAQUE = (0,   200, 255)
VERDE          = (0,   200, 100)
ROJO           = (200, 60,  60)
ACERO          = (150, 155, 160)
CIAN_AG        = (0,   255, 255)


class SalidaFotograma:
    """
    §9.1 — Gemelo digital 2D del rostro robótico EVA.

    Renderiza un rostro esquemático cuyos rasgos se mueven
    en función de las tensiones del mejor individuo del AG.

    Uso:
        salida = SalidaFotograma()
        salida.iniciar()
        salida.actualizar(tensiones=[0.3, 0.8, 0.8, 0.9, 0.9, 0.7], estado=...)
    """

    ANCHO = 480
    ALTO  = 480
    TITULO = "EVA - Gemelo Digital"

    def __init__(self):
        self.pantalla = None
        self.reloj = None
        self.fuente = None
        self.fuente_peq = None
        self._activa = False
        self.ver_motores = False  # Estado de visibilidad de actuadores
        self.rect_boton = pygame.Rect(320, 15, 145, 32)  # Área del botón
        self.config_ag = None
        self.material = None

    def set_parametros_bd(self, config_ag: dict, material: dict):
        """Asigna los parámetros de la Base de Datos para mostrarlos en la UI."""
        self.config_ag = config_ag
        self.material = material

    def iniciar(self) -> bool:
        """Inicializa la ventana Pygame."""
        pygame.init()
        self.pantalla = pygame.display.set_mode((self.ANCHO, self.ALTO))
        pygame.display.set_caption(self.TITULO)
        self.reloj = pygame.time.Clock()
        self.fuente = pygame.font.SysFont("monospace", 14, bold=True)
        self.fuente_peq = pygame.font.SysFont("monospace", 12)
        self._activa = True
        
        print(f"  🤖 Gemelo digital iniciado ({self.ANCHO}×{self.ALTO})")
        return True

    def actualizar(self, tensiones: List[float] = None,
                    estado: EstadoAG = None) -> bool:
        """
        Renderiza un fotograma del gemelo digital.

        Args:
            tensiones: Vector [m₁,...,m₆] del mejor individuo.
            estado:    Métricas del AG para el panel de etiquetas.

        Returns:
            False si el usuario cerró la ventana.
        """
        if not self._activa:
            return False

        # Procesar eventos de Pygame
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            if evento.type == pygame.KEYDOWN and evento.key == pygame.K_q:
                return False
            
            # Detectar clic en el botón de alternancia
            if evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                if self.rect_boton.collidepoint(evento.pos):
                    self.ver_motores = not self.ver_motores
                    print(f"  ⚙️ Visualización de motores: {'ON' if self.ver_motores else 'OFF'}")

        if tensiones is None:
            tensiones = [0.0] * 6
        if estado is None:
            estado = EstadoAG(tensiones=tensiones)

        # Limpiar pantalla
        self.pantalla.fill(FONDO)

        # Dibujar el rostro robótico
        self._dibujar_rostro(tensiones)

        # Dibujar arquitectura de motores si está activo
        if self.ver_motores:
            self._dibujar_actuadores(tensiones)

        # Dibujar panel de etiquetas (§9.1)
        self._dibujar_panel(estado)

        # Dibujar parámetros de BD en la esquina superior izquierda
        self._dibujar_parametros_bd()

        # Dibujar botón de alternancia
        self._dibujar_boton()

        pygame.display.flip()
        self.reloj.tick(30)
        return True

    def liberar(self):
        """Cierra la ventana Pygame."""
        self._activa = False
        pygame.quit()
        print("  🤖 Gemelo digital cerrado.")

    # ═══════════════════════════════════════════════════════════════
    #  RENDERIZADO DEL ROSTRO ROBÓTICO
    # ═══════════════════════════════════════════════════════════════

    def _dibujar_rostro(self, t: List[float]):
        """
        Dibuja el rostro esquemático del robot EVA.

        Mapeo de tensiones a rasgos:
            t[0] = m₁ (Boca/Mandíbula)  → apertura vertical de la boca
            t[1] = m₂ (Cigomático izq.) → elevación comisura izquierda
            t[2] = m₃ (Cigomático der.) → elevación comisura derecha
            t[3] = m₄ (Orbicular izq.)  → apertura párpado izquierdo
            t[4] = m₅ (Orbicular der.)  → apertura párpado derecho
            t[5] = m₆ (Frontal/Cejas)   → elevación de cejas
        """
        cx, cy = 240, 200  # Centro del rostro
        rect_cara = (cx - 100, cy - 120, 200, 260)

        # ── Contorno de la cabeza ───────────────────────────────
        pygame.draw.ellipse(self.pantalla, PIEL, rect_cara)
        pygame.draw.ellipse(self.pantalla, PIEL_OSCURA, rect_cara, 2)

        # ── Orejas ──────────────────────────────────────────────
        pygame.draw.ellipse(self.pantalla, PIEL, (cx - 115, cy - 30, 25, 50))
        pygame.draw.ellipse(self.pantalla, PIEL, (cx + 90, cy - 30, 25, 50))

        # ── Ojos ────────────────────────────────────────────────
        self._dibujar_ojo(cx - 45, cy - 20, t[3], espejo=False)  # Ojo izquierdo
        self._dibujar_ojo(cx + 45, cy - 20, t[4], espejo=True)   # Ojo derecho

        # ── Cejas ───────────────────────────────────────────────
        self._dibujar_cejas(cx, cy - 55, t[5])

        # ── Nariz ───────────────────────────────────────────────
        pygame.draw.line(self.pantalla, PIEL_OSCURA, (cx, cy + 5), (cx - 8, cy + 35), 2)
        pygame.draw.line(self.pantalla, PIEL_OSCURA, (cx - 8, cy + 35), (cx + 8, cy + 35), 2)

        # ── Boca ────────────────────────────────────────────────
        self._dibujar_boca(cx, cy + 65, t[0], t[1], t[2])

    def _dibujar_ojo(self, cx, cy, apertura, espejo=False):
        """
        Dibuja un ojo cuya apertura depende de la tensión del párpado.

        apertura = 0.0 → ojo cerrado (línea)
        apertura = 1.0 → ojo completamente abierto
        """
        ancho_ojo = 30
        alto_max = 18

        # Altura del ojo según apertura
        alto = max(2, int(alto_max * apertura))

        # Esclerótica (blanco del ojo)
        rect = pygame.Rect(cx - ancho_ojo, cy - alto // 2, ancho_ojo * 2, alto)
        pygame.draw.ellipse(self.pantalla, BLANCO_OJO, rect)
        pygame.draw.ellipse(self.pantalla, (50, 50, 60), rect, 2)

        # Iris y pupila (solo si el ojo está suficientemente abierto)
        if apertura > 0.15:
            radio_iris = min(8, alto // 2)
            pygame.draw.circle(self.pantalla, IRIS, (cx, cy), radio_iris)
            pygame.draw.circle(self.pantalla, PUPILA, (cx, cy), max(2, radio_iris // 2))

            # Brillo
            pygame.draw.circle(self.pantalla, (255, 255, 255),
                               (cx + 3, cy - 3), max(1, radio_iris // 4))

    def _dibujar_cejas(self, cx, cy, elevacion):
        """
        Dibuja las cejas. La elevación controla su posición vertical.

        elevacion = 0.0 → cejas bajas (fruncidas/enojado)
        elevacion = 1.0 → cejas altas (sorpresa)
        """
        desplazamiento = int(20 * (1.0 - elevacion))  # 0=arriba, 20=abajo

        # Ceja izquierda
        y_ceja = cy + desplazamiento
        puntos_izq = [
            (cx - 75, y_ceja + 5),
            (cx - 55, y_ceja - 5 + int(5 * (1 - elevacion))),
            (cx - 20, y_ceja)
        ]
        pygame.draw.lines(self.pantalla, CEJA, False, puntos_izq, 4)

        # Ceja derecha
        puntos_der = [
            (cx + 20, y_ceja),
            (cx + 55, y_ceja - 5 + int(5 * (1 - elevacion))),
            (cx + 75, y_ceja + 5)
        ]
        pygame.draw.lines(self.pantalla, CEJA, False, puntos_der, 4)

    def _dibujar_boca(self, cx, cy, apertura, comisura_izq, comisura_der):
        """
        Dibuja la boca del robot.

        apertura     = m₁ → apertura vertical (0=cerrada, 1=abierta)
        comisura_izq = m₂ → elevación comisura izquierda (0=caída, 1=sonrisa)
        comisura_der = m₃ → elevación comisura derecha
        """
        ancho_boca = 60
        alto_boca = max(2, int(30 * apertura))

        # Calcular curva de la sonrisa
        elev_izq = int(15 * (0.5 - comisura_izq))  # Positivo = baja (triste)
        elev_der = int(15 * (0.5 - comisura_der))

        # Puntos de la boca (curva Bézier simplificada con polilínea)
        puntos_superior = [
            (cx - ancho_boca, cy + elev_izq),
            (cx - ancho_boca // 2, cy - int(alto_boca * 0.3)),
            (cx, cy - int(alto_boca * 0.4)),
            (cx + ancho_boca // 2, cy - int(alto_boca * 0.3)),
            (cx + ancho_boca, cy + elev_der),
        ]

        puntos_inferior = [
            (cx - ancho_boca, cy + elev_izq),
            (cx - ancho_boca // 2, cy + int(alto_boca * 0.6)),
            (cx, cy + int(alto_boca * 0.8)),
            (cx + ancho_boca // 2, cy + int(alto_boca * 0.6)),
            (cx + ancho_boca, cy + elev_der),
        ]

        # Relleno de la boca si está abierta
        if apertura > 0.05:
            todos_puntos = puntos_superior + list(reversed(puntos_inferior))
            pygame.draw.polygon(self.pantalla, (60, 20, 20), todos_puntos)

        # Contorno de los labios
        pygame.draw.lines(self.pantalla, LABIO, False, puntos_superior, 3)
        pygame.draw.lines(self.pantalla, LABIO, False, puntos_inferior, 3)

    # ═══════════════════════════════════════════════════════════════
    #  ARQUITECTURA DE MOTORES Y TENSORES
    # ═══════════════════════════════════════════════════════════════

    def _dibujar_actuadores(self, t: List[float]):
        """
        Dibuja la arquitectura de tensores mecánicos sobre el rostro.
        Estilo: Tensores que tiran de cables y vectores de fuerza.
        """
        cx, cy = 240, 200
        
        # ── m6: Tensores Frontales (Cejas) ─────────────────────────
        # Anclajes en la parte superior del cráneo
        y_anc_ceja = cy - 100
        for i, offset in enumerate([-50, 50]):
            tension = t[5]
            # Punto de anclaje (motor)
            pygame.draw.circle(self.pantalla, ACERO, (cx + offset, y_anc_ceja), 6)
            # Cable/Tensor
            y_ceja_pos = cy - 55 + int(20 * (1.0 - tension))
            self._dibujar_tensor((cx + offset, y_anc_ceja), (cx + offset, y_ceja_pos), tension)

        # ── m4, m5: Tensores Orbiculares (Párpados) ────────────────
        y_anc_ojo = cy - 60
        pos_ojos = [(-45, 3), (45, 4)] # (offset_x, indice_tension)
        for offset_x, idx in pos_ojos:
            tension = t[idx]
            pygame.draw.circle(self.pantalla, ACERO, (cx + offset_x, y_anc_ojo), 5)
            # El cable tira del párpado superior
            alto_ojo = max(2, int(18 * tension))
            y_parpado = cy - 20 - alto_ojo // 2
            self._dibujar_tensor((cx + offset_x, y_anc_ojo), (cx + offset_x, y_parpado), tension)

        # ── m2, m3: Tensores Cigomáticos (Sonrisa) ─────────────────
        # Anclajes en los laterales (pómulos externos)
        y_anc_cig = cy + 10
        ancho_boca = 60
        # Tensiones m2 y m3
        pos_cig = [(-90, 1, -ancho_boca, 1), (90, 2, ancho_boca, 2)] # (x_anc, idx, x_boca, m_boca)
        for x_anc, idx, x_boca, m_idx in pos_cig:
            tension = t[idx]
            pygame.draw.circle(self.pantalla, ACERO, (cx + x_anc, y_anc_cig), 6)
            # Cable a la comisura
            elev = int(15 * (0.5 - t[m_idx]))
            self._dibujar_tensor((cx + x_anc, y_anc_cig), (cx + x_boca, cy + 65 + elev), tension)

        # ── m1: Tensor Mandibular (Boca) ───────────────────────────
        # Anclaje en la base de la barbilla, tira hacia abajo
        y_anc_man = cy + 170
        tension_man = t[0]
        pygame.draw.circle(self.pantalla, ACERO, (cx, y_anc_man), 8)
        alto_boca = max(2, int(30 * tension_man))
        self._dibujar_tensor((cx, y_anc_man), (cx, cy + 65 + int(alto_boca * 0.8)), tension_man)

    def _dibujar_tensor(self, inicio, fin, tension):
        """Dibuja un cable de acero y un vector de fuerza con brillo según tensión."""
        # Cable base
        pygame.draw.line(self.pantalla, (80, 80, 90), inicio, fin, 1)
        
        # Brillo del tensor según fuerza (Cian si hay mucha tensión)
        grosor = 1 + int(3 * tension)
        brillo = int(255 * tension)
        color = (0, brillo, brillo)
        if tension > 0.1:
            pygame.draw.line(self.pantalla, color, inicio, fin, grosor)
            
        # Cabeza del vector (flecha pequeña en el punto de tracción)
        if tension > 0.05:
            pygame.draw.circle(self.pantalla, CIAN_AG, fin, 3)

    def _dibujar_boton(self):
        """Dibuja el botón interactivo de Ver/Ocultar Motores."""
        mouse_pos = pygame.mouse.get_pos()
        hover = self.rect_boton.collidepoint(mouse_pos)
        
        color_fondo = (50, 50, 70) if not hover else (70, 70, 100)
        color_borde = CIAN_AG if self.ver_motores else (100, 100, 120)
        
        pygame.draw.rect(self.pantalla, color_fondo, self.rect_boton)
        pygame.draw.rect(self.pantalla, color_borde, self.rect_boton, 2)
        
        texto = "OCULTAR MOTORES" if self.ver_motores else "VER MOTORES"
        txt_surface = self.fuente_peq.render(texto, True, (255, 255, 255))
        rect_txt = txt_surface.get_rect(center=self.rect_boton.center)
        self.pantalla.blit(txt_surface, rect_txt)

    # ═══════════════════════════════════════════════════════════════
    #  PANEL DE ETIQUETAS (§9.1)
    # ═══════════════════════════════════════════════════════════════

    def _dibujar_panel(self, estado: EstadoAG):
        """
        §9.1 Panel de Etiquetas — Muestra métricas del AG superpuestas.
        """
        y = 340
        x = 10

        # Fondo del panel
        pygame.draw.rect(self.pantalla, PANEL_BG, (5, y - 5, 470, 135))
        pygame.draw.rect(self.pantalla, (60, 60, 80), (5, y - 5, 470, 135), 1)

        # Título
        self._texto(f"═══ SISTEMA EVA ═══", x + 140, y, TEXTO_DESTAQUE)
        y += 18

        # Métricas principales
        self._texto(f"Gen: {estado.generacion:>4}  |  "
                    f"Frame: {estado.fotograma:>5}  |  "
                    f"F: {estado.mejor_aptitud:.6f}  |  "
                    f"E: {estado.mejor_error:.6f}", x, y, TEXTO)
        y += 18

        # Tensiones del mejor individuo
        nombres = ['m₁', 'm₂', 'm₃', 'm₄', 'm₅', 'm₆']
        linea = "  ".join(f"{n}={v:.3f}" for n, v in zip(nombres, estado.tensiones))
        self._texto(f"Tensiones: {linea}", x, y, VERDE)
        y += 18

        # Barras visuales de tensión
        for i, (nombre, valor) in enumerate(zip(nombres, estado.tensiones)):
            bx = x + i * 76
            by = y + 5
            # Fondo
            pygame.draw.rect(self.pantalla, (40, 40, 50), (bx, by, 70, 10))
            # Barra rellena
            ancho = int(70 * valor)
            color = (int(255 * (1 - valor)), int(200 * valor), 80)
            pygame.draw.rect(self.pantalla, color, (bx, by, ancho, 10))
            # Etiqueta
            self._texto_peq(nombre, bx + 22, by + 18, TEXTO)

        y += 40

        # Razón de término (si existe)
        if estado.razon_termino:
            self._texto(f"Término: {estado.razon_termino}", x, y, ROJO)

    def _dibujar_parametros_bd(self):
        """Dibuja en pantalla los hiperparámetros cargados desde SQLite."""
        if not self.config_ag or not self.material:
            return
            
        x, y = 10, 10
        # Fondo oscuro del mini-panel
        pygame.draw.rect(self.pantalla, PANEL_BG, (5, 5, 145, 160))
        pygame.draw.rect(self.pantalla, (60, 60, 80), (5, 5, 145, 160), 1)
        
        # Hyperparameters del AG
        self._texto_peq("⚙️ CONFIG AG (BD)", x, y, TEXTO_DESTAQUE); y += 18
        self._texto_peq(f"N:      {self.config_ag.get('N')}", x, y, TEXTO); y += 14
        self._texto_peq(f"G_max:  {self.config_ag.get('G_max')}", x, y, TEXTO); y += 14
        self._texto_peq(f"pc:     {self.config_ag.get('pc')}", x, y, TEXTO); y += 14
        self._texto_peq(f"pm:     {self.config_ag.get('pm')}", x, y, TEXTO); y += 14
        self._texto_peq(f"Resol.: {self.config_ag.get('resolucion')}", x, y, TEXTO); y += 22
        
        # Parámetros del material
        self._texto_peq("🧪 MATERIAL (BD)", x, y, TEXTO_DESTAQUE); y += 18
        self._texto_peq(f"Amort.: {self.material.get('amortiguamiento')}", x, y, TEXTO); y += 14
        self._texto_peq(f"Elast.: {self.material.get('elasticidad')}", x, y, TEXTO); y += 14
        self._texto_peq(f"Límite: {self.material.get('limite_deformacion')}", x, y, TEXTO); y += 14

    def _texto(self, texto, x, y, color):
        """Helper para renderizar texto con la fuente principal."""
        superficie = self.fuente.render(texto, True, color)
        self.pantalla.blit(superficie, (x, y))

    def _texto_peq(self, texto, x, y, color):
        """Helper para renderizar texto con la fuente pequeña."""
        superficie = self.fuente_peq.render(texto, True, color)
        self.pantalla.blit(superficie, (x, y))


# ═══════════════════════════════════════════════════════════════════
#  EJECUCIÓN STANDALONE (prueba independiente de salida)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    import math

    salida = SalidaFotograma()
    salida.iniciar()

    print("  Demo del gemelo digital. Presiona 'q' o cierra la ventana para salir.\n")

    t_inicio = time.time()
    frame = 0

    try:
        while True:
            t = time.time() - t_inicio

            # Simular tensiones oscilantes para demostrar animación
            tensiones = [
                0.5 + 0.4 * math.sin(t * 2),            # m₁ Boca
                0.5 + 0.4 * math.sin(t * 1.5),           # m₂ CigIzq
                0.5 + 0.4 * math.sin(t * 1.5 + 0.2),     # m₃ CigDer
                0.5 + 0.4 * math.sin(t * 0.8),            # m₄ OjoIzq
                0.5 + 0.4 * math.sin(t * 0.8 + 0.1),      # m₅ OjoDer
                0.5 + 0.3 * math.sin(t * 1.2),            # m₆ Cejas
            ]
            tensiones = [max(0.0, min(1.0, v)) for v in tensiones]

            estado = EstadoAG(
                generacion=int(t * 5) % 50,
                fotograma=frame,
                mejor_aptitud=0.95 + 0.04 * math.sin(t),
                mejor_error=0.05 - 0.04 * math.sin(t),
                tensiones=tensiones,
            )

            if not salida.actualizar(tensiones, estado):
                break

            frame += 1

    except KeyboardInterrupt:
        pass
    finally:
        salida.liberar()
