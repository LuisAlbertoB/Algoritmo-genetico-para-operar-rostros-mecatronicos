#!/usr/bin/env python3
"""
launchOnlyAG.py
===============
Entry point para ejecutar SOLAMENTE el Algoritmo Genético
sin dependencias de cámara, MediaPipe ni Pygame.

Simula un vector objetivo T hardcodeado (una sonrisa) y ejecuta
el ciclo evolutivo completo, demostrando la convergencia del AG
hacia la expresión facial deseada.

Uso:
    python3 main/core/launchOnlyAG.py
"""

import sys
import os
import random

# Agregar el directorio AG al path para imports locales
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'AG'))

from genrarPoblacionInicial import (
    calcular_params_codificacion, generar_poblacion, imprimir_params
)
from evalucionDecodificarCromosoma import evaluar_poblacion
from cicloEvolutivo import ConfigAG, evolucionar_fotograma
from output import (
    calcular_estabilidad, calcular_diversidad, 
    imprimir_mejor_individuo
)


# ═══════════════════════════════════════════════════════════════════
#  VECTORES OBJETIVO DE PRUEBA (Expresiones faciales simuladas)
# ═══════════════════════════════════════════════════════════════════
#  Cada vector T = [t₁(Boca), t₂(CigI), t₃(CigD), 
#                   t₄(OjI), t₅(OjD), t₆(Cejas)]

EXPRESIONES_TEST = {
    'sonrisa': [0.30, 0.85, 0.85, 0.80, 0.80, 0.70],
    'sorpresa': [0.90, 0.10, 0.10, 0.95, 0.95, 0.95],
    'enojo': [0.10, 0.05, 0.05, 0.60, 0.60, 0.15],
    'neutral': [0.05, 0.10, 0.10, 0.85, 0.85, 0.50],
    'tristeza': [0.05, 0.02, 0.02, 0.70, 0.70, 0.30],
    'guiño': [0.20, 0.60, 0.30, 0.10, 0.90, 0.65],
}


def ejecutar_test(nombre_expresion: str, T: list, config: ConfigAG, params: dict):
    """Ejecuta el AG para una expresión facial objetivo."""
    
    print(f"\n{'#' * 65}")
    print(f"  EXPRESIÓN: {nombre_expresion.upper()}")
    print(f"  Target T = {[f'{t:.2f}' for t in T]}")
    print(f"{'#' * 65}\n")

    # §3 — Generar población inicial
    poblacion = generar_poblacion(config.N, params)

    # §13 — Ejecutar ciclo evolutivo completo
    resultado = evolucionar_fotograma(
        poblacion=poblacion,
        T=T,
        params=params,
        config=config,
        verbose=True
    )

    # Imprimir resultados
    print(f"\n  ── Razón de término: {resultado.razon_termino}")
    print(f"  ── Generaciones usadas: {resultado.generaciones_usadas}")

    imprimir_mejor_individuo(resultado.mejor)

    # Comparar target vs resultado
    print("\n  COMPARACIÓN TARGET vs RESULTADO:")
    print("  " + "-" * 55)
    nombres = ['Boca  ', 'CigIzq', 'CigDer', 'OjoIzq', 'OjoDer', 'Cejas ']
    for i, (nombre, t_val, m_val) in enumerate(zip(nombres, T, resultado.mejor.tensiones)):
        delta = abs(t_val - m_val)
        barra = '█' * int(delta * 100)
        print(f"  {nombre}: T={t_val:.4f} | M={m_val:.4f} | Δ={delta:.4f} {barra}")
    print("  " + "-" * 55)

    return resultado


def main():
    """Función principal del launcher de prueba del AG."""
    
    print("\n" + "═" * 65)
    print("  SISTEMA EVA — ALGORITMO GENÉTICO CANÓNICO (Solo Lógica)")
    print("  Fundamentos Matemáticos §1-§13")
    print("═" * 65)

    # ── Semilla para reproducibilidad ────────────────────────────
    random.seed(42)

    # ── §2.2 — Calcular parámetros de codificación binaria ──────
    params = calcular_params_codificacion()
    imprimir_params(params)

    # ── Configuración del AG (§13.2) ────────────────────────────
    config = ConfigAG(
        N=50,              # Tamaño de población
        k_torneo=3,        # Presión selectiva
        pc=0.8,            # Probabilidad de cruza
        tipo_cruza='1punto',
        pm=0.01,           # Probabilidad de mutación
        elites=2,          # Élites preservadas
        f_min=0.0,         # Sin umbral de poda
        G_max=50,          # Generaciones por fotograma (ampliado para demo)
        epsilon=0.001,     # Umbral de error aceptable
        w=5,               # Ventana de estancamiento
        sigma=0.0001,      # Umbral de estancamiento
    )

    print(f"\n  Configuración: N={config.N}, pc={config.pc}, pm={config.pm}, "
          f"G_max={config.G_max}, ε={config.epsilon}")

    # ── Ejecutar para cada expresión de prueba ──────────────────
    resultados = {}
    for nombre, T in EXPRESIONES_TEST.items():
        resultado = ejecutar_test(nombre, T, config, params)
        resultados[nombre] = resultado

    # ── Resumen final ───────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESUMEN DE TODAS LAS EXPRESIONES")
    print("═" * 65)
    print(f"  {'Expresión':<12} | {'Generaciones':>12} | {'Error MSE':>10} | {'Aptitud F':>10} | {'Razón'}")
    print("  " + "-" * 75)
    for nombre, res in resultados.items():
        print(f"  {nombre:<12} | {res.generaciones_usadas:>12} | "
              f"{res.mejor.error:>10.6f} | {res.mejor.aptitud:>10.6f} | "
              f"{res.razon_termino[:30]}")
    print("═" * 65)

    # ── Simulación de estabilidad entre 2 fotogramas ────────────
    print("\n  DEMO: Estabilidad inter-fotograma (§11.1)")
    print("  " + "-" * 50)
    if 'sonrisa' in resultados and 'sorpresa' in resultados:
        M_sonrisa = resultados['sonrisa'].mejor.tensiones
        M_sorpresa = resultados['sorpresa'].mejor.tensiones
        S = calcular_estabilidad(M_sorpresa, M_sonrisa)
        print(f"  S(sonrisa → sorpresa) = {S:.6f}")
        if S > 0.2:
            print("  ⚠ Movimiento brusco detectado (S > 0.2)")
        elif S > 0.05:
            print("  ✓ Transición aceptable (S > 0.05)")
        else:
            print("  ✓ Expresión estable (S ≈ 0)")

    print("\n  Ejecución completada exitosamente. ✓\n")


if __name__ == "__main__":
    main()
