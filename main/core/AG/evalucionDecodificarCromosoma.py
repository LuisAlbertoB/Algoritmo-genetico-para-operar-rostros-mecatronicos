"""
evalucionDecodificarCromosoma.py
================================
Módulo §2.4 y §4 de los Fundamentos Matemáticos.

Decodificación del cromosoma binario al fenotipo (vector de tensiones M),
cálculo del Error Cuadrático Medio (MSE) y la función de aptitud F.
"""

from typing import List
from genrarPoblacionInicial import Individuo


# ─── DECODIFICACIÓN (§2.4) ──────────────────────────────────────

def extraer_segmento(cromosoma: str, i: int, k: int = 10) -> str:
    """
    §2.4 Paso 1 — Extrae el segmento de k bits del actuador i.
    
    bits_i = cromosoma[(i-1)·k : i·k]
    
    Args:
        cromosoma: Cadena completa de L bits.
        i:         Índice del actuador (1-based: 1..6).
        k:         Bits por actuador (default: 10).
    
    Returns:
        Subcadena de k bits correspondiente al actuador i.
    """
    inicio = (i - 1) * k
    fin = i * k
    return cromosoma[inicio:fin]


def binario_a_decimal(cadena_bits: str) -> int:
    """
    §2.4 Paso 2 — Convierte una cadena binaria a valor decimal.
    
    d_i = Σ(j=0..k-1) b_j · 2^(k-1-j)
    """
    return int(cadena_bits, 2)


def decimal_a_real(d: int, a: float, rango: float, puntos_sistema: int) -> float:
    """
    §2.4 Paso 3 — Mapea un valor decimal al rango real [a, b].
    
    m_i = a + d_i · (R / (2^k - 1))
    
    Args:
        d:               Valor decimal decodificado.
        a:               Límite inferior del rango.
        rango:           R = b - a.
        puntos_sistema:  2^k (número de puntos representables).
    
    Returns:
        Valor real m_i ∈ [a, b].
    """
    return a + d * (rango / (puntos_sistema - 1))


def decodificar_cromosoma(cromosoma: str, params: dict) -> List[float]:
    """
    §2.4 — Pipeline completo de decodificación genotipo → fenotipo.
    
    Extrae las n subcadenas de k bits, las convierte a decimal,
    y las mapea al rango real [a, b] para obtener el vector M.
    
    Args:
        cromosoma: Cadena de L bits.
        params:    Diccionario de parámetros de codificación.
    
    Returns:
        Vector M = [m₁, m₂, ..., m₆] de tensiones reales en [0.0, 1.0].
    """
    n = params['n_actuadores']
    k = params['bits']
    a = params['a']
    rango = params['rango']
    p_sis = params['puntos_sistema']

    tensiones = []
    for i in range(1, n + 1):
        segmento = extraer_segmento(cromosoma, i, k)
        d = binario_a_decimal(segmento)
        m = decimal_a_real(d, a, rango, p_sis)
        tensiones.append(m)

    return tensiones


# ─── ERROR CUADRÁTICO MEDIO — MSE (§4.1) ────────────────────────

def calcular_mse(M: List[float], T: List[float]) -> float:
    """
    §4.1 — Calcula el Error Cuadrático Medio entre M y T.
    
    E(M, T) = (1/n) · Σ(i=1..n) (t_i - m_i)²
    
    Propiedades:
        - E ≥ 0 siempre.
        - E = 0 sii m_i = t_i ∀i (imitación perfecta).
        - E_max = 1.0 cuando todos los actuadores están en el extremo opuesto.
    
    Args:
        M: Vector de tensiones del individuo [m₁,...,m₆].
        T: Vector objetivo del usuario humano [t₁,...,t₆].
    
    Returns:
        Valor del MSE (float ≥ 0).
    """
    n = len(T)
    suma = sum((t - m) ** 2 for t, m in zip(T, M))
    return suma / n


# ─── FUNCIÓN DE APTITUD — FITNESS (§4.2) ────────────────────────

def calcular_aptitud(mse: float) -> float:
    """
    §4.2 — Transforma el error E en aptitud F mediante inversión hiperbólica.
    
    F(M, T) = 1 / (1 + E(M, T))
    
    Propiedades:
        - F ∈ (0, 1].
        - F = 1 cuando E = 0 (imitación perfecta → aptitud máxima).
        - F → 0 cuando E → ∞.
        - Monótonamente decreciente respecto a E.
    
    Args:
        mse: Error Cuadrático Medio (E ≥ 0).
    
    Returns:
        Valor de fitness F ∈ (0, 1].
    """
    return 1.0 / (1.0 + mse)


# ─── PIPELINE COMPLETO DE EVALUACIÓN ────────────────────────────

def evaluar_individuo(individuo: Individuo, T: List[float], params: dict) -> Individuo:
    """
    Pipeline completo: decodificar → calcular MSE → calcular fitness.
    
    Actualiza in-place los campos tensiones, error y aptitud del individuo.
    
    Args:
        individuo: Individuo con cromosoma ya asignado.
        T:         Vector objetivo [t₁,...,t₆].
        params:    Parámetros de codificación.
    
    Returns:
        El mismo individuo con sus campos actualizados.
    """
    individuo.tensiones = decodificar_cromosoma(individuo.cromosoma, params)
    individuo.error = calcular_mse(individuo.tensiones, T)
    individuo.aptitud = calcular_aptitud(individuo.error)
    return individuo


def evaluar_poblacion(poblacion: List[Individuo], T: List[float], params: dict) -> List[Individuo]:
    """
    §3.2 — Evalúa todos los individuos de la población.
    
    Args:
        poblacion: Lista de individuos.
        T:         Vector objetivo.
        params:    Parámetros de codificación.
    
    Returns:
        La misma lista con todos los individuos evaluados.
    """
    for ind in poblacion:
        evaluar_individuo(ind, T, params)
    return poblacion
