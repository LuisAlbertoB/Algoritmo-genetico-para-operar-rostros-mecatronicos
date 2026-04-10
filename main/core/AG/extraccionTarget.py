"""
extraccionTarget.py
===================
Módulo §1.2 de los Fundamentos Matemáticos.

Construye el vector objetivo T a partir de los blendshapes
detectados por MediaPipe, aplicando las inversiones semánticas
necesarias para alinear la medida de cierre visual con la
medida de apertura mecánica del actuador.
"""

from typing import Dict, List


# ─── Mapeo de Blendshapes a Actuadores (§1.2) ───────────────────

MAPEO_BLENDSHAPES = {
    # Componente  → (nombre_blendshape, invertir?)
    # t₁ = jawOpen                      → m₁ Mandibular
    # t₂ = mouthSmileLeft               → m₂ Cigomático izq.
    # t₃ = mouthSmileRight              → m₃ Cigomático der.
    # t₄ = 1.0 - eyeBlinkLeft           → m₄ Orbicular izq.
    # t₅ = 1.0 - eyeBlinkRight          → m₅ Orbicular der.
    # t₆ = 1.0 - browDownLeft           → m₆ Frontal
    't1': ('jawOpen', False),
    't2': ('mouthSmileLeft', False),
    't3': ('mouthSmileRight', False),
    't4': ('eyeBlinkLeft', True),      # Inversión: cierre → apertura
    't5': ('eyeBlinkRight', True),     # Inversión: cierre → apertura
    't6': ('browDownLeft', True),      # Inversión: fruncimiento → elevación
}

# Lista ordenada de los nombres de blendshape necesarios
BLENDSHAPES_REQUERIDOS = [
    'jawOpen', 'mouthSmileLeft', 'mouthSmileRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'browDownLeft'
]


def construir_vector_target(blendshapes: Dict[str, float]) -> List[float]:
    """
    §1.2 — Construye el vector objetivo T = [t₁, t₂, ..., t₆].
    
    Mapeo:
        t₁ = jawOpen                 (directo)
        t₂ = mouthSmileLeft          (directo)
        t₃ = mouthSmileRight         (directo)
        t₄ = 1.0 - eyeBlinkLeft     (invertido)
        t₅ = 1.0 - eyeBlinkRight    (invertido)
        t₆ = 1.0 - browDownLeft     (invertido)
    
    La inversión (1.0 - x) se debe a que MediaPipe reporta el
    "grado de cierre" (pestañeo/fruncimiento), pero el actuador
    robótico modela el "grado de apertura".
    
    Args:
        blendshapes: Diccionario {nombre_blendshape: valor} de MediaPipe.
                     Valores deben estar en [0.0, 1.0].
    
    Returns:
        Vector T = [t₁, ..., t₆], cada t_i ∈ [0.0, 1.0].
    """
    T = []

    for key in ['t1', 't2', 't3', 't4', 't5', 't6']:
        nombre, invertir = MAPEO_BLENDSHAPES[key]
        valor = blendshapes.get(nombre, 0.0)

        # Clampear al rango válido
        valor = max(0.0, min(1.0, valor))

        if invertir:
            valor = 1.0 - valor

        T.append(valor)

    return T


def construir_target_desde_lista(valores: List[float]) -> List[float]:
    """
    Construye un vector T directamente desde una lista ordenada de 6 valores.
    
    Útil para pruebas o cuando los valores ya están procesados.
    No aplica inversiones (asume que ya están aplicadas).
    
    Args:
        valores: Lista de 6 floats en [0.0, 1.0].
    
    Returns:
        Vector T clampeado a [0.0, 1.0].
    """
    return [max(0.0, min(1.0, v)) for v in valores]
