"""
cicloEvolutivo.py
=================
Módulo §10 y §13 de los Fundamentos Matemáticos.

Orquesta el ciclo evolutivo completo del AG para un fotograma,
integrando todos los operadores genéticos y las condiciones de término.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass, field

from genrarPoblacionInicial import Individuo, generar_poblacion, calcular_params_codificacion
from evalucionDecodificarCromosoma import evaluar_individuo, evaluar_poblacion
from seleccionTorneo import emparejar
from cruzamiento import cruza_un_punto, cruza_dos_puntos
from mutacion import mutar
from remplazoElitismoCombinarOrdenar import reemplazo_elitista
from poda import podar
from output import calcular_diversidad, imprimir_generacion


@dataclass
class ConfigAG:
    """
    Configuración del Algoritmo Genético (§13.2).
    Contiene todos los hiperparámetros del sistema.
    """
    # Población (§3)
    N: int                       # Tamaño de población
    
    # Selección (§5)
    k_torneo: int                # Tamaño del torneo
    
    # Cruzamiento (§6)
    pc: float                    # Probabilidad de cruza
    tipo_cruza: str              # '1punto' o '2puntos'
    
    # Mutación (§7)
    pm: float                    # Probabilidad mutación por bit
    
    # Elitismo (§8)
    elites: int                  # Número de élites
    
    # Poda (§9)
    f_min: float                 # Umbral mínimo de aptitud
    
    # Condiciones de término (§10)
    G_max: int                   # Máximo generaciones por fotograma
    epsilon: float               # Umbral de error aceptable
    w: int                       # Ventana de estancamiento
    sigma: float                 # Umbral de estancamiento


@dataclass
class ResultadoFotograma:
    """Resultado del ciclo evolutivo para un fotograma."""
    mejor: Individuo = None
    poblacion: List[Individuo] = field(default_factory=list)
    generaciones_usadas: int = 0
    historial_mejor_F: List[float] = field(default_factory=list)
    historial_promedio_F: List[float] = field(default_factory=list)
    historial_diversidad: List[float] = field(default_factory=list)
    razon_termino: str = ""


def evolucionar_fotograma(poblacion: List[Individuo],
                           T: List[float],
                           params: dict,
                           config: ConfigAG,
                           verbose: bool = True) -> ResultadoFotograma:
    """
    §13.1 — Ciclo evolutivo completo para un fotograma.
    
    Orquesta los pasos 5-11 del diagrama de flujo:
        5. Evaluación
        6. Condición de término
        7. Selección (Torneo)
        8. Cruzamiento
        9. Mutación
        10. Reemplazo + Elitismo
        11. Poda
    
    Condiciones de término (§10.1):
        Criterio 1: g ≥ G_max
        Criterio 2: E(M_mejor, T) < ε
        Criterio 3: |F_max^(g) - F_max^(g-w)| < σ (estancamiento)
    
    Args:
        poblacion: Población actual (puede ser la inicial o heredada).
        T:         Vector objetivo [t₁,...,t₆].
        params:    Parámetros de codificación binaria.
        config:    Configuración obligatoria del AG proveniente de la BD.
        verbose:   Si True, imprime progreso por consola.
    
    Returns:
        ResultadoFotograma con el mejor individuo, población final y métricas.
    """
    if config is None:
        raise ValueError("Es totalmente obligatorio proveer los hiperparámetros desde un ConfigAG validado.")

    resultado = ResultadoFotograma()

    # ── Paso 5: Evaluación inicial ──────────────────────────────
    evaluar_poblacion(poblacion, T, params)

    # Seleccionar función de cruza
    func_cruza = cruza_un_punto if config.tipo_cruza == '1punto' else cruza_dos_puntos

    for g in range(config.G_max):
        # ── Estadísticas de la generación actual ────────────────
        aptitudes = [ind.aptitud for ind in poblacion]
        mejor_actual = max(poblacion, key=lambda ind: ind.aptitud)
        promedio = sum(aptitudes) / len(aptitudes)
        diversidad = calcular_diversidad(poblacion, params['longitud_cromosoma'])

        resultado.historial_mejor_F.append(mejor_actual.aptitud)
        resultado.historial_promedio_F.append(promedio)
        resultado.historial_diversidad.append(diversidad)

        if verbose:
            imprimir_generacion(g + 1, mejor_actual, promedio, diversidad)

        # ── Paso 6: Condición de término ────────────────────────

        # Criterio 2: Umbral de error alcanzado (§10.1)
        if mejor_actual.error < config.epsilon:
            resultado.razon_termino = f"Criterio 2: E={mejor_actual.error:.6f} < ε={config.epsilon}"
            resultado.mejor = mejor_actual
            resultado.generaciones_usadas = g + 1
            resultado.poblacion = poblacion
            return resultado

        # Criterio 3: Estancamiento evolutivo (§10.1)
        if g >= config.w:
            f_actual = resultado.historial_mejor_F[-1]
            f_anterior = resultado.historial_mejor_F[-1 - config.w]
            if abs(f_actual - f_anterior) < config.sigma:
                resultado.razon_termino = (
                    f"Criterio 3: Estancamiento ΔF={abs(f_actual - f_anterior):.8f} < σ={config.sigma}")
                resultado.mejor = mejor_actual
                resultado.generaciones_usadas = g + 1
                resultado.poblacion = poblacion
                return resultado

        # ── Paso 7: Selección por Torneo ────────────────────────
        parejas = emparejar(poblacion, config.k_torneo)

        # ── Paso 8: Cruzamiento ─────────────────────────────────
        hijos = []
        for padre1, padre2 in parejas:
            hijo1, hijo2 = func_cruza(padre1, padre2, config.pc)
            hijos.append(hijo1)
            hijos.append(hijo2)

        # ── Paso 9: Mutación ────────────────────────────────────
        hijos = [mutar(hijo, config.pm) for hijo in hijos]

        # Evaluar hijos
        evaluar_poblacion(hijos, T, params)

        # ── Paso 10: Reemplazo + Elitismo ───────────────────────
        poblacion = reemplazo_elitista(poblacion, hijos, config.elites)

        # ── Paso 11: Poda ───────────────────────────────────────
        poblacion = podar(poblacion, config.N, config.f_min, params)

        # Los nuevos individuos inyectados por poda necesitan evaluación
        for ind in poblacion:
            if ind.aptitud == 0.0 and ind.error == float('inf'):
                evaluar_individuo(ind, T, params)

    # Criterio 1: Máximo de generaciones alcanzado (§10.1)
    mejor_final = max(poblacion, key=lambda ind: ind.aptitud)
    resultado.razon_termino = f"Criterio 1: g={config.G_max} ≥ G_max={config.G_max}"
    resultado.mejor = mejor_final
    resultado.generaciones_usadas = config.G_max
    resultado.poblacion = poblacion

    return resultado
