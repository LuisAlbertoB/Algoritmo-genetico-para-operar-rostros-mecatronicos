"""
seleccionTorneo.py
==================
Módulo §5 de los Fundamentos Matemáticos.

Implementa la selección por torneo estocástico y el emparejamiento
de progenitores para el operador de cruzamiento.
"""

import random
from typing import List, Tuple
from genrarPoblacionInicial import Individuo


def seleccion_torneo(poblacion: List[Individuo], k: int = 3) -> Individuo:
    """
    §5.2 — Selecciona un individuo mediante torneo de tamaño k.
    
    Algoritmo:
        1. Elegir aleatoriamente k individuos de P_g.
        2. Comparar la aptitud F de los k competidores.
        3. El individuo con mayor F gana el torneo.
    
    Formalización:
        Padre = argmax_{I ∈ S_k} F(I)
        donde S_k ⊂ P_g es un subconjunto aleatorio de tamaño k.
    
    Presión selectiva (§5.3):
        k=2 → Presión baja (más diversidad, convergencia lenta)
        k=3 → Presión equilibrada (recomendado)
        k=5 → Presión alta (convergencia rápida, riesgo óptimos locales)
        k=N → Determinístico (siempre gana el mejor)
    
    Args:
        poblacion: Lista de individuos evaluados.
        k:         Tamaño del torneo (default: 3).
    
    Returns:
        El individuo ganador del torneo.
    """
    competidores = random.sample(poblacion, min(k, len(poblacion)))
    return max(competidores, key=lambda ind: ind.aptitud)


def emparejar(poblacion: List[Individuo], k: int = 3) -> List[Tuple[Individuo, Individuo]]:
    """
    §5 — Genera N/2 parejas de progenitores mediante selección por torneo.
    
    Cada padre se selecciona de forma independiente mediante un torneo
    separado. Es posible que el mismo individuo sea seleccionado
    como padre en múltiples parejas (selección con reemplazo implícito).
    
    Args:
        poblacion: Lista de individuos evaluados.
        k:         Tamaño del torneo para la selección.
    
    Returns:
        Lista de tuplas (padre1, padre2).
    """
    num_parejas = len(poblacion) // 2
    parejas = []

    for _ in range(num_parejas):
        padre1 = seleccion_torneo(poblacion, k)
        padre2 = seleccion_torneo(poblacion, k)
        parejas.append((padre1, padre2))

    return parejas
