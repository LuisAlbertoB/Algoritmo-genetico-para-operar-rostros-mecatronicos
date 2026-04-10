"""
mutacion.py
===========
Módulo §7 de los Fundamentos Matemáticos.

Implementa la mutación por inversión de bit (bit-flip) con
probabilidad configurable por locus.
"""

import random
from genrarPoblacionInicial import Individuo


def mutar(individuo: Individuo, pm: float = 0.01) -> Individuo:
    """
    §7.1 — Mutación por inversión de bit (Bit-Flip).
    
    Para cada bit b_l del cromosoma (l = 1, ..., L):
        b_l' = 1 - b_l   si r_l < p_m
        b_l' = b_l        si r_l ≥ p_m
    donde r_l ~ Uniforme(0, 1).
    
    §7.2 — Número esperado de bits mutados:
        E[mutaciones] = L · p_m = 60 · 0.01 = 0.6 bits
        (menos de un bit por individuo por generación en promedio).
    
    §7.4 — Impacto posicional:
        Bit 1 (MSB): Δm_i ≈ 50% (exploración agresiva)
        Bit 5 (medio): Δm_i ≈ 3.1% (ajuste gestual)
        Bit 10 (LSB): Δm_i ≈ 0.1% (micro-pulsación)
    
    Args:
        individuo: Individuo cuyo cromosoma será mutado.
        pm:        Probabilidad de mutación por bit (default: 0.01).
    
    Returns:
        Nuevo Individuo con el cromosoma potencialmente mutado.
    """
    bits = list(individuo.cromosoma)

    for i in range(len(bits)):
        if random.random() < pm:
            # Inversión del bit: '0' → '1', '1' → '0'
            bits[i] = '1' if bits[i] == '0' else '0'

    nuevo_cromosoma = ''.join(bits)
    return Individuo(cromosoma=nuevo_cromosoma)
