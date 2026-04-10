"""
genrarPoblacionInicial.py
=========================
Módulo §2 y §3 de los Fundamentos Matemáticos.

Define la estructura de datos del Individuo, los parámetros de
codificación binaria y la generación de la población inicial
con distribución Bernoulli(0.5).
"""

import math
import random
from dataclasses import dataclass, field
from typing import List





@dataclass
class Individuo:
    """
    Representa un individuo (solución candidata) en la población del AG.
    
    Atributos:
        cromosoma:  Cadena de L=60 bits (genotipo).
        tensiones:  Vector M = [m₁,...,m₆] decodificado del cromosoma (fenotipo).
        error:      Error Cuadrático Medio E(M, T) respecto al target actual.
        aptitud:    Fitness F = 1/(1+E). Valor a maximizar por el AG.
    """
    cromosoma: str = ""
    tensiones: List[float] = field(default_factory=list)
    error: float = float('inf')
    aptitud: float = 0.0

    def __str__(self) -> str:
        tens_str = ", ".join(f"{t:.4f}" for t in self.tensiones)
        return (f"Cromosoma: {self.cromosoma[:20]}... | "
                f"M=[{tens_str}] | E={self.error:.6f} | F={self.aptitud:.6f}")


def calcular_params_codificacion(a: float, b: float, delta: float, n_actuadores: int) -> dict:
    """
    §2.2 — Calcula los parámetros del sistema de codificación binaria.
    
    Fórmulas:
        R = b - a
        P_problema = ⌊R/δ⌋ + 1
        k = ⌈log₂(P_problema)⌉
        P_sistema = 2^k
        δ_real = R / (P_sistema - 1)
    
    Returns:
        dict con: a, b, rango, bits, puntos_problema, puntos_sistema,
                  resolucion_real, n_actuadores, longitud_cromosoma
    """
    rango = b - a
    puntos_problema = int(rango / delta) + 1
    bits = math.ceil(math.log2(puntos_problema))
    puntos_sistema = 2 ** bits
    resolucion_real = rango / (puntos_sistema - 1)

    return {
        'a': a,
        'b': b,
        'rango': rango,
        'bits': bits,
        'puntos_problema': puntos_problema,
        'puntos_sistema': puntos_sistema,
        'resolucion_real': resolucion_real,
        'n_actuadores': n_actuadores,
        'longitud_cromosoma': n_actuadores * bits,
    }


def generar_cromosoma_aleatorio(longitud: int) -> str:
    """
    §3.1 — Genera un cromosoma de L bits con distribución Bernoulli(0.5).
    
    Cada bit b_l ~ Bernoulli(0.5): 50% probabilidad de ser '0' o '1'.
    """
    return ''.join(random.choice('01') for _ in range(longitud))


def generar_poblacion(N: int, params: dict) -> List[Individuo]:
    """
    §3.1 — Genera la población inicial P₀ de N individuos aleatorios.
    
    Cada individuo se genera como un cromosoma aleatorio.
    La decodificación y evaluación se delega a evalucionDecodificarCromosoma.py
    para respetar la separación modular.
    
    Args:
        N:      Tamaño de la población (recomendado: 30-200, default: 50).
        params: Diccionario de parámetros de codificación.
    
    Returns:
        Lista de N individuos con cromosomas aleatorios (sin evaluar aún).
    """
    L = params['longitud_cromosoma']
    poblacion = []

    for _ in range(N):
        cromosoma = generar_cromosoma_aleatorio(L)
        individuo = Individuo(cromosoma=cromosoma)
        poblacion.append(individuo)

    return poblacion


def imprimir_params(params: dict) -> None:
    """Imprime los parámetros del sistema de codificación en consola."""
    print("=" * 60)
    print("PARÁMETROS DEL SISTEMA DE CODIFICACIÓN BINARIA")
    print("=" * 60)
    print(f"  Rango [a, b]           : [{params['a']}, {params['b']}]")
    print(f"  R = b - a              : {params['rango']}")
    print(f"  Puntos problema        : {params['puntos_problema']}")
    print(f"  Bits por actuador (k)  : {params['bits']}")
    print(f"  Puntos sistema (2^k)   : {params['puntos_sistema']}")
    print(f"  Resolución real        : {params['resolucion_real']:.6f}")
    print(f"  Actuadores (n)         : {params['n_actuadores']}")
    print(f"  Longitud cromosoma (L) : {params['longitud_cromosoma']} bits")
    print("=" * 60)
