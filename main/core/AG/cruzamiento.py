"""
cruzamiento.py
==============
Módulo §6 de los Fundamentos Matemáticos.

Implementa los operadores de cruza de un punto y dos puntos
para la recombinación de cromosomas parentales.
"""

import random
from typing import Tuple
from genrarPoblacionInicial import Individuo


def cruza_un_punto(padre1: Individuo, padre2: Individuo, 
                    pc: float = 0.8) -> Tuple[Individuo, Individuo]:
    """
    §6.1 — Cruza de un punto entre dos cromosomas parentales.
    
    Paso 1: c ~ Uniforme{1, 2, ..., L-1}
    Paso 2: H₁ = P₁[1:c] ⊕ P₂[c+1:L]
            H₂ = P₂[1:c] ⊕ P₁[c+1:L]
    
    §6.3 — La cruza se ejecuta con probabilidad p_c.
            Si r ≥ p_c, los hijos son copias de los padres.
    
    §6.4 — Significado biológico:
        Si c cae en fronteras (10, 20, 30, 40, 50) → hereda músculos completos.
        Si c cae intra-gen → recombinación fina (tensiones híbridas).
    
    Args:
        padre1: Primer progenitor.
        padre2: Segundo progenitor.
        pc:     Probabilidad de cruza (default: 0.8).
    
    Returns:
        Tupla de dos hijos (Individuo) con cromosomas recombinados.
    """
    # §6.3 — Verificar si la cruza ocurre
    if random.random() >= pc:
        # No hay cruza: hijos son copias de los padres
        hijo1 = Individuo(cromosoma=padre1.cromosoma)
        hijo2 = Individuo(cromosoma=padre2.cromosoma)
        return hijo1, hijo2

    L = len(padre1.cromosoma)
    
    # Paso 1: Punto de corte aleatorio
    c = random.randint(1, L - 1)

    # Paso 2: Crear hijos por concatenación de segmentos
    hijo1_bits = padre1.cromosoma[:c] + padre2.cromosoma[c:]
    hijo2_bits = padre2.cromosoma[:c] + padre1.cromosoma[c:]

    hijo1 = Individuo(cromosoma=hijo1_bits)
    hijo2 = Individuo(cromosoma=hijo2_bits)

    return hijo1, hijo2


def cruza_dos_puntos(padre1: Individuo, padre2: Individuo,
                      pc: float = 0.8) -> Tuple[Individuo, Individuo]:
    """
    §6.2 — Cruza de dos puntos: intercambia el segmento central.
    
    Puntos de corte:
        c₁ ~ Uniforme{1, ..., L-2}
        c₂ ~ Uniforme{c₁+1, ..., L-1}
    
    Hijos:
        H₁ = P₁[1:c₁] ⊕ P₂[c₁+1:c₂] ⊕ P₁[c₂+1:L]
        H₂ = P₂[1:c₁] ⊕ P₁[c₁+1:c₂] ⊕ P₂[c₂+1:L]
    
    Args:
        padre1: Primer progenitor.
        padre2: Segundo progenitor.
        pc:     Probabilidad de cruza (default: 0.8).
    
    Returns:
        Tupla de dos hijos con segmento central intercambiado.
    """
    if random.random() >= pc:
        hijo1 = Individuo(cromosoma=padre1.cromosoma)
        hijo2 = Individuo(cromosoma=padre2.cromosoma)
        return hijo1, hijo2

    L = len(padre1.cromosoma)

    c1 = random.randint(1, L - 2)
    c2 = random.randint(c1 + 1, L - 1)

    hijo1_bits = padre1.cromosoma[:c1] + padre2.cromosoma[c1:c2] + padre1.cromosoma[c2:]
    hijo2_bits = padre2.cromosoma[:c1] + padre1.cromosoma[c1:c2] + padre2.cromosoma[c2:]

    hijo1 = Individuo(cromosoma=hijo1_bits)
    hijo2 = Individuo(cromosoma=hijo2_bits)

    return hijo1, hijo2
