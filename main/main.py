#!/usr/bin/env python3
"""
main.py
=======
Lanzador Unificado del Sistema EVA.

Ejecuta simultáneamente:
    - Base de Conocimiento (SQLite)              → Configuración + Registro
    - Interfaz de Entrada (OpenCV + MediaPipe)    → Vector T
    - Modelo de Material (Silicona)               → Transformación física
    - Núcleo del AG (Ciclo Evolutivo)             → Tensiones M
    - Interfaz de Salida  (Pygame Gemelo 2D)      → Visualización

Uso:
    python3 main/main.py

Controles:
    q → Salir del sistema
"""

import sys
import os
import time
import random

# ── Configurar paths de importación ─────────────────────────────
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MAIN_DIR, 'client'))
sys.path.insert(0, os.path.join(MAIN_DIR, 'core', 'AG'))
sys.path.insert(0, os.path.join(MAIN_DIR, 'core', 'infraestructure'))

from interfaz import InterfazEntrada
from salidaDelFotograma import SalidaFotograma, EstadoAG
from genrarPoblacionInicial import calcular_params_codificacion, generar_poblacion
from cicloEvolutivo import ConfigAG, evolucionar_fotograma
from db import BaseConocimiento
from materialSimulacion import ModeloMaterial
from reportes import GeneradorReportes


# ── Constante: frecuencia de registro en BD (cada ~1 segundo) ───
FRAMES_POR_REGISTRO = 15


def main():
    print("\n" + "═" * 60)
    print("  SISTEMA EVA — ALGORITMO GENÉTICO EN TIEMPO REAL")
    print("  Entrada: Cámara + MediaPipe")
    print("  Salida:  Gemelo Digital 2D (Pygame)")
    print("  Datos:   SQLite + Modelo de Material")
    print("═" * 60)

    # ── Semilla para reproducibilidad ────────────────────────────
    random.seed(42)

    # ═══════════════════════════════════════════════════════════════
    #  FASE 1: Base de Conocimiento (§7.1)
    # ═══════════════════════════════════════════════════════════════

    bc = BaseConocimiento()
    bc.conectar()
    bc.crear_schema()
    bc.poblar_conocimiento()

    # ── Cargar ConfigAG desde la BD ──────────────────────────────
    config_dict = bc.cargar_config_ag()
    config = ConfigAG(
        N=config_dict.get('N'),
        k_torneo=config_dict.get('k_torneo'),
        pc=config_dict.get('pc'),
        pm=config_dict.get('pm'),
        elites=config_dict.get('elites'),
        f_min=config_dict.get('f_min', 0.0), # Respaldo inicial en caso BD vieja
        G_max=config_dict.get('G_max'),
        epsilon=config_dict.get('epsilon'),
        w=config_dict.get('w'),
        sigma=config_dict.get('sigma'),
        tipo_cruza=config_dict.get('tipo_cruza'),
    )

    print(f"\n  AG Config (desde BD): N={config.N}, G_max={config.G_max}, "
          f"ε={config.epsilon}, pc={config.pc}, pm={config.pm}")

    # ── Cargar Modelo de Material ────────────────────────────────
    params_material = bc.cargar_parametros_material()
    modelo_material = ModeloMaterial(params_material)
    print(f"  Material: {modelo_material}")

    # ── Registrar nueva sesión ───────────────────────────────────
    sesion_id = bc.registrar_sesion(config_dict)

    # ═══════════════════════════════════════════════════════════════
    #  FASE 2: Inicializar Módulos
    # ═══════════════════════════════════════════════════════════════

    robot_modelos = bc.cargar_robot_modelo()
    n_actuadores = len(robot_modelos)
    rango_min = min([r['rango_min'] for r in robot_modelos]) if robot_modelos else 0.0
    rango_max = max([r['rango_max'] for r in robot_modelos]) if robot_modelos else 1.0
    resolucion = config_dict.get('resolucion', 0.001)

    params = calcular_params_codificacion(
        a=rango_min,
        b=rango_max,
        delta=resolucion,
        n_actuadores=n_actuadores
    )

    entrada = InterfazEntrada(
        indice_camara=0,
        ancho=320, alto=240, fps=30,
        confianza=0.5
    )
    salida = SalidaFotograma()
    salida.set_parametros_bd(config_dict, params_material)

    if not entrada.iniciar():
        print("  ⚠ Error al iniciar la interfaz de entrada.")
        bc.cerrar()
        return

    if not salida.iniciar():
        print("  ⚠ Error al iniciar el gemelo digital.")
        entrada.liberar()
        bc.cerrar()
        return

    # ── Población inicial ────────────────────────────────────────
    poblacion = generar_poblacion(config.N, params)

    print("\n  ✓ Sistema listo. Presiona 'q' para salir.\n")

    # ═══════════════════════════════════════════════════════════════
    #  FASE 3: Bucle Principal
    # ═══════════════════════════════════════════════════════════════

    fotograma = 0
    tensiones_actuales = [0.0] * 6
    tensiones_material = [0.0] * 6  # Después del modelo de material
    mejor_aptitud = 0.0
    mejor_error = float('inf')
    generacion_actual = 0
    razon_termino = ""
    T = [0.0] * 6

    # Para el podio de mejores individuos
    top3_sesion = []  # Lista de (aptitud, individuo_data, fotograma)

    t_inicio = time.time()

    try:
        while True:
            # ── ENTRADA: Captura + Detección + Vector T ─────────
            frame_viz, T, detectado = entrada.procesar_frame()

            if frame_viz is None:
                break

            # ── AG: Ciclo Evolutivo (solo si hay rostro) ────────
            if detectado and any(t > 0.01 for t in T):
                resultado = evolucionar_fotograma(
                    poblacion=poblacion,
                    T=T,
                    params=params,
                    config=config,
                    verbose=False
                )

                tensiones_actuales = resultado.mejor.tensiones
                mejor_aptitud = resultado.mejor.aptitud
                mejor_error = resultado.mejor.error
                generacion_actual = resultado.generaciones_usadas
                razon_termino = resultado.razon_termino
                poblacion = resultado.poblacion

                # ── MATERIAL: Aplicar modelo de silicona ────────
                tensiones_material = modelo_material.transformar_tensiones(
                    tensiones_actuales
                )

                # ── Actualizar podio de mejores ─────────────────
                individuo_data = {
                    'cromosoma': resultado.mejor.cromosoma,
                    **{f'm{i+1}': v for i, v in enumerate(tensiones_actuales)},
                    'aptitud': mejor_aptitud,
                    'error': mejor_error,
                    'generacion': generacion_actual,
                    'fotograma': fotograma,
                }
                top3_sesion.append((mejor_aptitud, individuo_data))
                # Mantener solo top 3
                top3_sesion.sort(key=lambda x: x[0], reverse=True)
                top3_sesion = top3_sesion[:3]

            fotograma += 1

            # ── SALIDA: Actualizar gemelo digital ───────────────
            # El gemelo recibe las tensiones transformadas por el material
            estado = EstadoAG(
                generacion=generacion_actual,
                fotograma=fotograma,
                mejor_aptitud=mejor_aptitud,
                mejor_error=mejor_error,
                tensiones=tensiones_material,
                razon_termino=razon_termino,
            )

            if not salida.actualizar(tensiones_material, estado):
                break

            # ── ENTRADA: Mostrar ventana OpenCV ─────────────────
            if not entrada.mostrar(frame_viz):
                break

            # ── BD: Registrar fotograma (cada ~1 segundo) ───────
            if fotograma % FRAMES_POR_REGISTRO == 0:
                datos_frame = {
                    'numero': fotograma,
                    'mejor_F': mejor_aptitud,
                    'mejor_E': mejor_error,
                    'F_promedio': mejor_aptitud * 0.95,  # Aproximación
                    'diversidad': 0.3,  # Se calculará del resultado
                    **{f't{i+1}': v for i, v in enumerate(T)},
                    **{f'm{i+1}': v for i, v in enumerate(tensiones_material)},
                }

                # Añadir diversidad del resultado si disponible
                if detectado and hasattr(resultado, 'historial_diversidad') and resultado.historial_diversidad:
                    datos_frame['diversidad'] = resultado.historial_diversidad[-1]
                if detectado and hasattr(resultado, 'historial_promedio_F') and resultado.historial_promedio_F:
                    datos_frame['F_promedio'] = resultado.historial_promedio_F[-1]

                bc.registrar_fotograma(sesion_id, datos_frame)

            # ── Telemetría periódica en consola ─────────────────
            if fotograma % 30 == 0:
                dt_total = time.time() - t_inicio
                fps_avg = fotograma / dt_total if dt_total > 0 else 0
                t_str = " ".join(f"{v:.2f}" for v in T)
                m_str = " ".join(f"{v:.2f}" for v in tensiones_material)
                print(f"  Frame {fotograma:>5} | "
                      f"FPS:{fps_avg:.1f} | "
                      f"F:{mejor_aptitud:.4f} | "
                      f"E:{mejor_error:.4f} | "
                      f"T=[{t_str}] → M'=[{m_str}]")

    except KeyboardInterrupt:
        print("\n  Interrupción de teclado.")

    finally:
        # ═══════════════════════════════════════════════════════════
        #  FASE 4: Cierre y Reportes
        # ═══════════════════════════════════════════════════════════

        dt_total = time.time() - t_inicio
        fps_avg = fotograma / dt_total if dt_total > 0 else 0

        print(f"\n  ── Resumen de sesión ──")
        print(f"  Fotogramas procesados: {fotograma}")
        print(f"  Duración total: {dt_total:.1f}s")
        print(f"  FPS promedio: {fps_avg:.1f}")
        print(f"  Última aptitud: F={mejor_aptitud:.6f}")
        print(f"  Último error:   E={mejor_error:.6f}")

        # Commit final de fotogramas pendientes
        bc.commit()

        # Registrar podio de mejores individuos
        if top3_sesion:
            mejores_para_bd = []
            for pos, (apt, data) in enumerate(top3_sesion, 1):
                data['posicion'] = pos
                mejores_para_bd.append(data)
            bc.registrar_mejores(sesion_id, mejores_para_bd)

        # Cerrar sesión en BD
        bc.cerrar_sesion(sesion_id, fotograma, fps_avg)

        # Generar reportes automáticos (§9.2)
        try:
            gen = GeneradorReportes(bc.conn)
            gen.generar_reporte_completo(sesion_id)
        except Exception as e:
            print(f"  ⚠ Error al generar reportes: {e}")

        # Liberar recursos
        entrada.liberar()
        salida.liberar()
        bc.cerrar()
        print("  ✓ Sistema EVA finalizado.\n")


if __name__ == "__main__":
    main()
