"""
remplazoElitismoCombinarOrdenar.py
===================================
Módulo §8 de los Fundamentos Matemáticos.

Implementa el reemplazo generacional con preservación elitista,
combinación de poblaciones y ordenamiento por aptitud.
"""

from typing import List
from genrarPoblacionInicial import Individuo


def reemplazo_elitista(poblacion: List[Individuo], 
                        hijos: List[Individuo], 
                        e: int = 2) -> List[Individuo]:
    """
    §8.2 — Algoritmo de Reemplazo Elitista en 3 pasos.
    
    Paso 1: Preservar la élite.
        Elite_g = Top_e(P_g)
    
    Paso 2: Combinar con la nueva generación.
        P_{g+1}^{comb} = Elite_g ∪ Hijos_g
    
    Paso 3: Seleccionar los N mejores para la nueva generación.
        P_{g+1} = Top_N(P_{g+1}^{comb})
    
    Garantiza que los mejores individuos NUNCA se pierdan
    entre generaciones, protegiendo los máximos temporales.
    
    Args:
        poblacion:  Población actual P_g (ya evaluada).
        hijos:      Hijos generados por cruza + mutación (ya evaluados).
        e:          Número de élites a preservar (default: 2).
    
    Returns:
        Nueva población P_{g+1} de tamaño N.
    """
    N = len(poblacion)

    # Paso 1 — Preservar la élite
    poblacion_ordenada = sorted(poblacion, key=lambda ind: ind.aptitud, reverse=True)
    elites = poblacion_ordenada[:e]

    # Paso 2 — Combinar
    combinada = elites + hijos

    # Paso 3 — Ordenar y truncar a N
    combinada_ordenada = sorted(combinada, key=lambda ind: ind.aptitud, reverse=True)

    return combinada_ordenada[:N]
