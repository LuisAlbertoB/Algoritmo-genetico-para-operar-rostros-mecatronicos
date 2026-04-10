"""
output.py
=========
Módulo §11 de los Fundamentos Matemáticos.

Implementa las métricas de diagnóstico del sistema:
- Estabilidad inter-fotograma (S)
- Diversidad poblacional de Hamming (D)
- Funciones de impresión por consola
"""

from typing import List
from genrarPoblacionInicial import Individuo


# ─── ESTABILIDAD ENTRE FOTOGRAMAS (§11.1) ────────────────────────

def calcular_estabilidad(M_actual: List[float], 
                          M_anterior: List[float]) -> float:
    """
    §11.1 — Estabilidad cinética inter-fotograma.
    
    S^(f) = (1/n) · Σ(i=1..n) (m_i^(f) - m_i^(f-1))²
    
    Interpretación:
        S ≈ 0     → Expresión estable (ideal en sonrisa sostenida).
        S > 0.05  → Transición rápida (aceptable al cambiar de gesto).
        S > 0.2   → Espasmos (probable convergencia prematura).
    
    Args:
        M_actual:   Vector [m₁,...,m₆] del mejor individuo en fotograma f.
        M_anterior: Vector [m₁,...,m₆] del mejor individuo en fotograma f-1.
    
    Returns:
        Valor de estabilidad S ≥ 0.
    """
    n = len(M_actual)
    suma = sum((a - b) ** 2 for a, b in zip(M_actual, M_anterior))
    return suma / n


# ─── DIVERSIDAD POBLACIONAL POR HAMMING (§11.2) ─────────────────

def distancia_hamming(crom1: str, crom2: str) -> int:
    """
    §11.2 — Distancia de Hamming entre dos cromosomas.
    
    d_H(I_i, I_j) = Σ(l=1..L) 𝟙[b_l^(i) ≠ b_l^(j)]
    
    Es el número de posiciones de bits en las que difieren.
    """
    return sum(c1 != c2 for c1, c2 in zip(crom1, crom2))


def calcular_diversidad(poblacion: List[Individuo], L: int = 60) -> float:
    """
    §11.2 — Diversidad poblacional normalizada.
    
    D^(g) = (2 / (N(N-1))) · Σ(i=1..N-1) Σ(j=i+1..N) d_H(I_i, I_j)
    D_norm = D^(g) / L
    
    Interpretación:
        D_norm < 0.1         → Homogénea (riesgo convergencia prematura).
        0.2 ≤ D_norm ≤ 0.4   → Rango saludable (equilibrio).
        D_norm > 0.5         → Demasiado dispersa (no converge).
    
    Args:
        poblacion: Lista de individuos con cromosomas asignados.
        L:         Longitud del cromosoma (default: 60).
    
    Returns:
        Diversidad normalizada D_norm ∈ [0, 1].
    """
    N = len(poblacion)

    if N < 2:
        return 0.0

    suma_hamming = 0.0
    pares = 0

    for i in range(N - 1):
        for j in range(i + 1, N):
            suma_hamming += distancia_hamming(
                poblacion[i].cromosoma, 
                poblacion[j].cromosoma
            )
            pares += 1

    # D promedio
    D = suma_hamming / pares if pares > 0 else 0.0

    # Normalizar por la longitud del cromosoma
    return D / L


# ─── FUNCIONES DE IMPRESIÓN ──────────────────────────────────────

def imprimir_generacion(gen: int, mejor: Individuo, 
                         promedio_aptitud: float, diversidad: float) -> None:
    """Imprime el resumen de una generación en consola."""
    print(f"  Gen {gen:>3} | "
          f"Mejor F={mejor.aptitud:.6f} | "
          f"E={mejor.error:.6f} | "
          f"F_prom={promedio_aptitud:.6f} | "
          f"D={diversidad:.4f}")


def imprimir_mejor_individuo(mejor: Individuo) -> None:
    """Imprime un reporte detallado del mejor individuo."""
    print("\n" + "=" * 65)
    print("  MEJOR INDIVIDUO ENCONTRADO")
    print("=" * 65)
    print(f"  Cromosoma : {mejor.cromosoma}")
    print(f"  Tensiones : ", end="")
    nombres = ['m₁(Boca)', 'm₂(CigI)', 'm₃(CigD)', 
                'm₄(OjI)', 'm₅(OjD)', 'm₆(Cejas)']
    for nombre, valor in zip(nombres, mejor.tensiones):
        print(f"{nombre}={valor:.4f}  ", end="")
    print()
    print(f"  Error MSE : {mejor.error:.8f}")
    print(f"  Aptitud F : {mejor.aptitud:.8f}")
    print("=" * 65)
