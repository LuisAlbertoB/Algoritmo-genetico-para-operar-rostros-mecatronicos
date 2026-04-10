"""
reportes.py
===========
§9.2 — Salidas Estáticas (Bajo Demanda).

Genera gráficas y reportes visuales al finalizar una sesión:
    - Evolución de Aptitud (F_max y F_prom)
    - Decaimiento del Error (MSE)
    - Diversidad Poblacional
    - Tabla Podio de los Mejores Individuos
    - Validación Cruzada K-fold por Emoción

Las gráficas se guardan como PNG en temp/reportes/.
"""

import os
import math
import sqlite3
from typing import Optional, List, Dict, Tuple

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sin ventana para producción
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB_DISPONIBLE = True
except ImportError:
    MATPLOTLIB_DISPONIBLE = False


# Ruta de salida para los reportes
_MODULO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROYECTO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_MODULO_DIR)))
RUTA_REPORTES = os.path.join(_PROYECTO_DIR, 'temp', 'reportes')


class GeneradorReportes:
    """
    §9.2 — Generador de reportes visuales del sistema EVA.

    Uso:
        gen = GeneradorReportes(conn)
        gen.generar_reporte_completo(sesion_id)
    """

    # Paleta de colores premium
    COLOR_FONDO = '#1e1e2e'
    COLOR_TEXTO = '#cdd6f4'
    COLOR_GRID  = '#45475a'
    COLOR_LINEA_1 = '#89b4fa'  # Azul
    COLOR_LINEA_2 = '#f38ba8'  # Rosa
    COLOR_LINEA_3 = '#a6e3a1'  # Verde
    COLOR_RELLENO = '#313244'

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        os.makedirs(RUTA_REPORTES, exist_ok=True)

    def _estilo_base(self, fig, ax, titulo: str, xlabel: str, ylabel: str):
        """Aplica el estilo visual base a una gráfica."""
        fig.patch.set_facecolor(self.COLOR_FONDO)
        ax.set_facecolor(self.COLOR_FONDO)
        ax.set_title(titulo, color=self.COLOR_TEXTO, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(xlabel, color=self.COLOR_TEXTO, fontsize=11)
        ax.set_ylabel(ylabel, color=self.COLOR_TEXTO, fontsize=11)
        ax.tick_params(colors=self.COLOR_TEXTO, labelsize=9)
        ax.grid(True, alpha=0.3, color=self.COLOR_GRID, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(self.COLOR_GRID)
        ax.spines['left'].set_color(self.COLOR_GRID)

    def grafica_evolucion_aptitud(self, sesion_id: int) -> Optional[str]:
        """
        §9.2 — Gráfica de Evolución de Aptitud.
        Curvas F_max y F_prom por fotograma.
        """
        if not MATPLOTLIB_DISPONIBLE:
            print("  ⚠ matplotlib no disponible. Saltando gráfica de aptitud.")
            return None

        c = self.conn.cursor()
        c.execute("""
            SELECT numero, mejor_F, F_promedio
            FROM fotogramas WHERE sesion_id = ?
            ORDER BY numero
        """, (sesion_id,))
        datos = c.fetchall()

        if not datos:
            return None

        frames = [d[0] for d in datos]
        f_max = [d[1] for d in datos]
        f_prom = [d[2] for d in datos]

        fig, ax = plt.subplots(figsize=(12, 6))
        self._estilo_base(fig, ax, 
                          f'Evolución de Aptitud — Sesión #{sesion_id}',
                          'Fotograma', 'Aptitud F')

        ax.plot(frames, f_max, color=self.COLOR_LINEA_1,
                linewidth=2, label='F_max (mejor)', alpha=0.9)
        ax.plot(frames, f_prom, color=self.COLOR_LINEA_2,
                linewidth=1.5, label='F_prom (promedio)', alpha=0.7)
        ax.fill_between(frames, f_prom, f_max, 
                         alpha=0.15, color=self.COLOR_LINEA_1)
        ax.legend(facecolor=self.COLOR_FONDO, edgecolor=self.COLOR_GRID,
                  labelcolor=self.COLOR_TEXTO, fontsize=10)

        ruta = os.path.join(RUTA_REPORTES, f'aptitud_sesion_{sesion_id}.png')
        fig.savefig(ruta, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  📊 Gráfica de aptitud: {ruta}")
        return ruta

    def grafica_decaimiento_error(self, sesion_id: int) -> Optional[str]:
        """
        §9.2 — Gráfica de Decaimiento del Error (MSE).
        """
        if not MATPLOTLIB_DISPONIBLE:
            return None

        c = self.conn.cursor()
        c.execute("""
            SELECT numero, mejor_E FROM fotogramas
            WHERE sesion_id = ? ORDER BY numero
        """, (sesion_id,))
        datos = c.fetchall()

        if not datos:
            return None

        frames = [d[0] for d in datos]
        errores = [d[1] for d in datos]

        fig, ax = plt.subplots(figsize=(12, 6))
        self._estilo_base(fig, ax,
                          f'Decaimiento del Error (MSE) — Sesión #{sesion_id}',
                          'Fotograma', 'Error E (MSE)')

        ax.plot(frames, errores, color=self.COLOR_LINEA_2,
                linewidth=2, alpha=0.9)
        ax.fill_between(frames, errores, alpha=0.2, color=self.COLOR_LINEA_2)

        # Línea de umbral ε
        ax.axhline(y=0.005, color=self.COLOR_LINEA_3,
                   linestyle='--', linewidth=1, alpha=0.6, label='ε = 0.005')
        ax.legend(facecolor=self.COLOR_FONDO, edgecolor=self.COLOR_GRID,
                  labelcolor=self.COLOR_TEXTO)

        ruta = os.path.join(RUTA_REPORTES, f'error_sesion_{sesion_id}.png')
        fig.savefig(ruta, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  📊 Gráfica de error: {ruta}")
        return ruta

    def grafica_diversidad(self, sesion_id: int) -> Optional[str]:
        """
        §9.2 — Gráfica de Diversidad Poblacional.
        """
        if not MATPLOTLIB_DISPONIBLE:
            return None

        c = self.conn.cursor()
        c.execute("""
            SELECT numero, diversidad FROM fotogramas
            WHERE sesion_id = ? ORDER BY numero
        """, (sesion_id,))
        datos = c.fetchall()

        if not datos:
            return None

        frames = [d[0] for d in datos]
        diversidades = [d[1] for d in datos]

        fig, ax = plt.subplots(figsize=(12, 6))
        self._estilo_base(fig, ax,
                          f'Diversidad Poblacional — Sesión #{sesion_id}',
                          'Fotograma', 'Diversidad D (normalizada)')

        ax.plot(frames, diversidades, color=self.COLOR_LINEA_3,
                linewidth=2, alpha=0.9)
        ax.fill_between(frames, diversidades, alpha=0.15, color=self.COLOR_LINEA_3)

        # Bandas de referencia
        ax.axhspan(0.2, 0.4, alpha=0.08, color=self.COLOR_LINEA_3, label='Rango saludable')
        ax.axhline(y=0.1, color=self.COLOR_LINEA_2,
                   linestyle=':', linewidth=1, alpha=0.5, label='Riesgo convergencia prematura')
        ax.legend(facecolor=self.COLOR_FONDO, edgecolor=self.COLOR_GRID,
                  labelcolor=self.COLOR_TEXTO, fontsize=9)

        ruta = os.path.join(RUTA_REPORTES, f'diversidad_sesion_{sesion_id}.png')
        fig.savefig(ruta, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  📊 Gráfica de diversidad: {ruta}")
        return ruta

    def tabla_mejores(self, sesion_id: int) -> Optional[str]:
        """
        §9.2 — Tabla Podio de los 3 Mejores Individuos.
        """
        if not MATPLOTLIB_DISPONIBLE:
            return None

        c = self.conn.cursor()
        c.execute("""
            SELECT posicion, m1,m2,m3,m4,m5,m6, aptitud, error, generacion, fotograma
            FROM mejores_individuos
            WHERE sesion_id = ?
            ORDER BY posicion
        """, (sesion_id,))
        datos = c.fetchall()

        if not datos:
            return None

        # Crear tabla visual
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor(self.COLOR_FONDO)
        ax.set_facecolor(self.COLOR_FONDO)
        ax.axis('off')

        medallas = ['🥇', '🥈', '🥉']
        headers = ['Pos', 'm₁', 'm₂', 'm₃', 'm₄', 'm₅', 'm₆', 'Aptitud F', 'Error E', 'Gen', 'Frame']

        tabla_datos = []
        for d in datos:
            pos = d[0]
            fila = [
                medallas[pos - 1] if pos <= 3 else str(pos),
                f'{d[1]:.3f}', f'{d[2]:.3f}', f'{d[3]:.3f}',
                f'{d[4]:.3f}', f'{d[5]:.3f}', f'{d[6]:.3f}',
                f'{d[7]:.6f}', f'{d[8]:.6f}',
                str(d[9]), str(d[10])
            ]
            tabla_datos.append(fila)

        tabla = ax.table(cellText=tabla_datos, colLabels=headers,
                         cellLoc='center', loc='center')
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(1, 1.8)

        # Estilo de celdas
        for (i, j), celda in tabla.get_celda_dict().items() if hasattr(tabla, 'get_celda_dict') else []:
            celda.set_facecolor(self.COLOR_FONDO)
            celda.set_text_props(color=self.COLOR_TEXTO)
            celda.set_edgecolor(self.COLOR_GRID)

        # Fallback: estilo por método estándar
        for key, cell in tabla.get_celld().items():
            cell.set_facecolor(self.COLOR_FONDO if key[0] > 0 else '#313244')
            cell.set_text_props(color=self.COLOR_TEXTO)
            cell.set_edgecolor(self.COLOR_GRID)

        ax.set_title(f'Podio de Mejores Individuos — Sesión #{sesion_id}',
                     color=self.COLOR_TEXTO, fontsize=14, fontweight='bold', pad=20)

        ruta = os.path.join(RUTA_REPORTES, f'podio_sesion_{sesion_id}.png')
        fig.savefig(ruta, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  📊 Tabla podio: {ruta}")
        return ruta

    # ═══════════════════════════════════════════════════════════════
    #  §9.2 — VALIDACIÓN CRUZADA K-FOLD POR EMOCIÓN
    # ═══════════════════════════════════════════════════════════════

    def _clasificar_emocion(self, t: List[float],
                             ekman: List[Dict]) -> str:
        """
        Clasifica un vector Target T en la emoción de Ekman más cercana
        usando distancia euclidiana.
        """
        mejor_emocion = 'Desconocida'
        mejor_dist = float('inf')

        for emo in ekman:
            vec_emo = [emo['m1'], emo['m2'], emo['m3'],
                       emo['m4'], emo['m5'], emo['m6']]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(t, vec_emo)))
            if dist < mejor_dist:
                mejor_dist = dist
                mejor_emocion = emo['nombre_emocion']

        return mejor_emocion

    def validacion_cruzada(self, sesion_id: int,
                            K: int = 5) -> Optional[str]:
        """
        §9.2 — Reporte de Validación Cruzada K-fold.

        Clasifica cada fotograma registrado en la emoción de Ekman
        más cercana a su vector Target, luego realiza una partición
        K-fold para evaluar si el AG generaliza bien entre diferentes
        tipos de expresiones o si sobreajusta a un solo gesto.

        Genera una gráfica con:
            - Barras de error promedio por emoción (train vs test)
            - Matriz de confusión emoción × fold
            - Estadísticas de generalización

        Args:
            sesion_id: ID de la sesión a analizar.
            K: Número de folds (por defecto 5).
        """
        if not MATPLOTLIB_DISPONIBLE:
            print("  ⚠ matplotlib no disponible. Saltando validación cruzada.")
            return None

        c = self.conn.cursor()

        # ── Cargar fotogramas con target y error ─────────────────
        c.execute("""
            SELECT t1,t2,t3,t4,t5,t6, mejor_E
            FROM fotogramas WHERE sesion_id = ?
            ORDER BY numero
        """, (sesion_id,))
        datos_raw = c.fetchall()

        if not datos_raw or len(datos_raw) < K:
            print(f"  ⚠ Datos insuficientes para K={K} folds ({len(datos_raw)} fotogramas).")
            return None

        # ── Cargar expresiones de Ekman de referencia ────────────
        c.execute("SELECT nombre_emocion, m1,m2,m3,m4,m5,m6 FROM expresiones_ekman")
        ekman = [dict(zip(['nombre_emocion','m1','m2','m3','m4','m5','m6'], row))
                 for row in c.fetchall()]

        if not ekman:
            print("  ⚠ No hay expresiones de Ekman en la BD.")
            return None

        # ── Clasificar cada fotograma en una emoción ─────────────
        fotogramas_clasificados = []
        for row in datos_raw:
            t_vec = [row[0], row[1], row[2], row[3], row[4], row[5]]
            error = row[6]
            emocion = self._clasificar_emocion(t_vec, ekman)
            fotogramas_clasificados.append({
                'emocion': emocion,
                'error': error,
                'target': t_vec,
            })

        # ── Agrupar por emoción ──────────────────────────────────
        emociones_presentes = sorted(set(f['emocion'] for f in fotogramas_clasificados))
        errores_por_emocion = {}
        for emo in emociones_presentes:
            errores_por_emocion[emo] = [
                f['error'] for f in fotogramas_clasificados if f['emocion'] == emo
            ]

        # ── Partición K-fold ─────────────────────────────────────
        n_total = len(fotogramas_clasificados)
        tam_fold = n_total // K
        resultados_fold = []  # Lista de dicts {fold, E_train, E_test}

        for k in range(K):
            inicio = k * tam_fold
            fin = inicio + tam_fold if k < K - 1 else n_total

            fold_test = fotogramas_clasificados[inicio:fin]
            fold_train = fotogramas_clasificados[:inicio] + fotogramas_clasificados[fin:]

            e_train = sum(f['error'] for f in fold_train) / len(fold_train) if fold_train else 0
            e_test = sum(f['error'] for f in fold_test) / len(fold_test) if fold_test else 0

            resultados_fold.append({
                'fold': k + 1,
                'E_train': e_train,
                'E_test': e_test,
                'n_test': len(fold_test),
                'n_train': len(fold_train),
            })

        # ── Estadísticas globales ────────────────────────────────
        e_test_global = [r['E_test'] for r in resultados_fold]
        e_train_global = [r['E_train'] for r in resultados_fold]
        media_test = sum(e_test_global) / len(e_test_global)
        media_train = sum(e_train_global) / len(e_train_global)
        varianza_test = sum((e - media_test) ** 2 for e in e_test_global) / len(e_test_global)
        desv_test = math.sqrt(varianza_test)
        gap = abs(media_test - media_train)

        # ── GRÁFICA: Figura compuesta (2 subplots) ──────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                         gridspec_kw={'width_ratios': [1.2, 1]})
        fig.patch.set_facecolor(self.COLOR_FONDO)
        fig.suptitle(f'Validación Cruzada K-Fold (K={K}) — Sesión #{sesion_id}',
                     color=self.COLOR_TEXTO, fontsize=16, fontweight='bold', y=0.98)

        # ── Subplot 1: Error por Fold (Train vs Test) ────────────
        ax1.set_facecolor(self.COLOR_FONDO)
        folds = [r['fold'] for r in resultados_fold]
        bars_train = [r['E_train'] for r in resultados_fold]
        bars_test = [r['E_test'] for r in resultados_fold]

        x_pos = list(range(len(folds)))
        ancho = 0.35
        x_train = [p - ancho / 2 for p in x_pos]
        x_test = [p + ancho / 2 for p in x_pos]

        ax1.bar(x_train, bars_train, ancho, label='E_train',
                color=self.COLOR_LINEA_1, alpha=0.85, edgecolor='none')
        ax1.bar(x_test, bars_test, ancho, label='E_test',
                color=self.COLOR_LINEA_2, alpha=0.85, edgecolor='none')

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'Fold {f}' for f in folds], color=self.COLOR_TEXTO)
        ax1.set_ylabel('Error Promedio (MSE)', color=self.COLOR_TEXTO, fontsize=11)
        ax1.set_title('Error por Fold', color=self.COLOR_TEXTO, fontsize=13, pad=10)
        ax1.tick_params(colors=self.COLOR_TEXTO, labelsize=9)
        ax1.grid(True, alpha=0.2, color=self.COLOR_GRID, linestyle='--', axis='y')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(self.COLOR_GRID)
        ax1.spines['left'].set_color(self.COLOR_GRID)
        ax1.legend(facecolor=self.COLOR_FONDO, edgecolor=self.COLOR_GRID,
                   labelcolor=self.COLOR_TEXTO, fontsize=10)

        # ── Subplot 2: Error promedio por Emoción ────────────────
        ax2.set_facecolor(self.COLOR_FONDO)

        emo_nombres = list(errores_por_emocion.keys())
        emo_medias = [sum(v) / len(v) for v in errores_por_emocion.values()]
        emo_counts = [len(v) for v in errores_por_emocion.values()]

        colores_emo = [self.COLOR_LINEA_1, self.COLOR_LINEA_2,
                       self.COLOR_LINEA_3, '#fab387', '#cba6f7', '#f9e2af']
        colores_usados = [colores_emo[i % len(colores_emo)] for i in range(len(emo_nombres))]

        barras = ax2.barh(emo_nombres, emo_medias, color=colores_usados,
                          alpha=0.85, edgecolor='none', height=0.6)

        # Etiquetas con conteo de fotogramas
        for i, (barra, count) in enumerate(zip(barras, emo_counts)):
            ax2.text(barra.get_width() + 0.0003, barra.get_y() + barra.get_height() / 2,
                     f'  n={count}', va='center', color=self.COLOR_TEXTO, fontsize=9)

        ax2.set_xlabel('Error Promedio (MSE)', color=self.COLOR_TEXTO, fontsize=11)
        ax2.set_title('Error por Emoción Detectada', color=self.COLOR_TEXTO, fontsize=13, pad=10)
        ax2.tick_params(colors=self.COLOR_TEXTO, labelsize=10)
        ax2.grid(True, alpha=0.2, color=self.COLOR_GRID, linestyle='--', axis='x')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_color(self.COLOR_GRID)
        ax2.spines['left'].set_color(self.COLOR_GRID)
        ax2.invert_yaxis()

        # ── Caja de estadísticas ─────────────────────────────────
        sobreajuste = 'Sí' if gap > 0.005 else 'No'
        stats_text = (
            f"Estadísticas de Generalización\n"
            f"{'─' * 32}\n"
            f"E_train promedio:  {media_train:.6f}\n"
            f"E_test  promedio:  {media_test:.6f}\n"
            f"Desv. estándar:    {desv_test:.6f}\n"
            f"Gap (test-train):  {gap:.6f}\n"
            f"Sobreajuste:       {sobreajuste}\n"
            f"Fotogramas total:  {n_total}\n"
            f"Emociones detectadas: {len(emociones_presentes)}"
        )
        fig.text(0.5, -0.02, stats_text, ha='center', va='top',
                 fontsize=10, color=self.COLOR_TEXTO,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.8',
                           facecolor='#313244', edgecolor=self.COLOR_GRID,
                           alpha=0.9))

        plt.tight_layout(rect=[0, 0.08, 1, 0.94])

        ruta = os.path.join(RUTA_REPORTES, f'validacion_cruzada_sesion_{sesion_id}.png')
        fig.savefig(ruta, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  📊 Validación cruzada K-fold: {ruta}")
        return ruta

    # ═══════════════════════════════════════════════════════════════
    #  GENERADOR DE REPORTE COMPLETO
    # ═══════════════════════════════════════════════════════════════

    def generar_reporte_completo(self, sesion_id: int) -> dict:
        """
        §9.2 — Genera todas las gráficas y las guarda como PNG.

        Returns:
            dict con las rutas de todos los archivos generados.
        """
        print(f"\n  ══ Generando Reportes — Sesión #{sesion_id} ══")

        rutas = {
            'aptitud': self.grafica_evolucion_aptitud(sesion_id),
            'error': self.grafica_decaimiento_error(sesion_id),
            'diversidad': self.grafica_diversidad(sesion_id),
            'podio': self.tabla_mejores(sesion_id),
            'validacion_cruzada': self.validacion_cruzada(sesion_id),
        }

        generados = sum(1 for v in rutas.values() if v is not None)
        print(f"  ══ {generados} reportes generados en {RUTA_REPORTES} ══\n")

        return rutas


# ═══════════════════════════════════════════════════════════════════
#  PRUEBA STANDALONE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from db import BaseConocimiento

    bc = BaseConocimiento()
    bc.conectar()
    bc.crear_schema()
    bc.poblar_conocimiento()

    # Crear sesión de prueba con datos sintéticos
    sesion_id = bc.registrar_sesion({'N': 30, 'test': True})

    import random
    for i in range(100):
        f_val = 0.90 + 0.09 * (i / 100) + random.uniform(-0.01, 0.01)
        e_val = max(0, 0.10 - 0.09 * (i / 100) + random.uniform(-0.005, 0.005))
        datos = {
            'numero': i + 1,
            'mejor_F': f_val, 'mejor_E': e_val,
            'F_promedio': f_val - 0.03,
            'diversidad': 0.35 - 0.15 * (i / 100) + random.uniform(-0.02, 0.02),
            **{f't{j}': random.uniform(0, 1) for j in range(1, 7)},
            **{f'm{j}': random.uniform(0, 1) for j in range(1, 7)},
        }
        bc.registrar_fotograma(sesion_id, datos)
    bc.commit()

    # Registrar podio
    bc.registrar_mejores(sesion_id, [
        {'posicion': 1, 'm1': 0.85, 'm2': 0.90, 'm3': 0.88, 'm4': 0.75, 'm5': 0.78, 'm6': 0.60,
         'aptitud': 0.9821, 'error': 0.0179, 'generacion': 34, 'fotograma': 95},
        {'posicion': 2, 'm1': 0.82, 'm2': 0.87, 'm3': 0.85, 'm4': 0.72, 'm5': 0.74, 'm6': 0.58,
         'aptitud': 0.9743, 'error': 0.0257, 'generacion': 31, 'fotograma': 88},
        {'posicion': 3, 'm1': 0.80, 'm2': 0.89, 'm3': 0.83, 'm4': 0.70, 'm5': 0.76, 'm6': 0.55,
         'aptitud': 0.9695, 'error': 0.0305, 'generacion': 28, 'fotograma': 80},
    ])

    bc.cerrar_sesion(sesion_id, 100, 14.9)

    # Generar reportes
    gen = GeneradorReportes(bc.conn)
    gen.generar_reporte_completo(sesion_id)

    bc.cerrar()
