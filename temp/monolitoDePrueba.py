#!/usr/bin/env python3
"""
monolitoDePrueba.py
===================
Versión monolítica (archivo único) del Algoritmo Genético para EVA.
Consolida la lógica de 10 módulos separados en un solo script
para facilitar la experimentación rápida.

Basado en los "Fundamentos Matemáticos" (§1 a §13).
"""

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silencia logs de TF
import tensorflow as tf
from sklearn.model_selection import KFold


# ═══════════════════════════════════════════════════════════════════
#  1. CONSTANTES DEL PROBLEMA Y CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════

N_ACTUADORES = 6          # n = 6 actuadores faciales
BITS_POR_ACTUADOR = 10    # k = 10 bits por actuador
LONGITUD_CROMOSOMA = N_ACTUADORES * BITS_POR_ACTUADOR  # L = 60 bits
RANGO_MIN = 0.0           # a = 0.0
RANGO_MAX = 1.0           # b = 1.0
RESOLUCION_DESEADA = 1e-3 # δ = 10⁻³

@dataclass
class ConfigAG:
    """Configuración del Algoritmo Genético (§13.2)."""
    N: int = 50                  # Tamaño de población
    k_torneo: int = 3            # Tamaño del torneo
    pc: float = 0.8              # Probabilidad de cruza
    tipo_cruza: str = '1punto'   # '1punto' o '2puntos'
    pm: float = 0.01             # Probabilidad mutación por bit
    elites: int = 2              # Número de élites
    f_min: float = 0.0           # Umbral mínimo de aptitud
    G_max: int = 50              # Máximo generaciones por fotograma
    epsilon: float = 0.001       # Umbral de error aceptable
    w: int = 5                   # Ventana de estancamiento
    sigma: float = 0.0001        # Umbral de estancamiento


# ═══════════════════════════════════════════════════════════════════
#  2. ESTRUCTURA BASE Y POBLACIÓN (§2, §3)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Individuo:
    """Representa un individuo de la población del AG."""
    cromosoma: str = ""
    tensiones: List[float] = field(default_factory=list)
    error: float = float('inf')
    aptitud: float = 0.0

def calcular_params_codificacion(a=RANGO_MIN, b=RANGO_MAX, delta=RESOLUCION_DESEADA) -> dict:
    """§2.2 — Parámetros matemáticos del sistema binario."""
    rango = b - a
    puntos_problema = int(rango / delta) + 1
    bits = math.ceil(math.log2(puntos_problema))
    p_sis = 2 ** bits
    res_real = rango / (p_sis - 1)
    return {
        'a': a, 'b': b, 'rango': rango, 'bits': bits,
        'puntos_problema': puntos_problema, 'puntos_sistema': p_sis,
        'resolucion_real': res_real, 'n_actuadores': N_ACTUADORES,
        'longitud_cromosoma': N_ACTUADORES * bits,
    }

def generar_cromosoma_aleatorio(L: int = LONGITUD_CROMOSOMA) -> str:
    """§3.1 — Genera cromosoma de L bits con Bernoulli(0.5)."""
    return ''.join(random.choice('01') for _ in range(L))

def generar_poblacion(N: int, params: dict) -> List[Individuo]:
    """§3.1 — Genera P₀."""
    return [Individuo(cromosoma=generar_cromosoma_aleatorio(params['longitud_cromosoma'])) 
            for _ in range(N)]


# ═══════════════════════════════════════════════════════════════════
#  3. DECODIFICACIÓN Y EVALUACIÓN (§2.4, §4)
# ═══════════════════════════════════════════════════════════════════

def decodificar_cromosoma(cromosoma: str, params: dict) -> List[float]:
    """§2.4 — Convierte 60 bits a 6 tensiones [0.0, 1.0]."""
    k = params['bits']
    a = params['a']
    rango = params['rango']
    p_sis = params['puntos_sistema']
    tensiones = []
    
    for i in range(1, params['n_actuadores'] + 1):
        bits_i = cromosoma[(i-1)*k : i*k]
        d_i = int(bits_i, 2)
        m_i = a + d_i * (rango / (p_sis - 1))
        tensiones.append(m_i)
    return tensiones

def evaluar_individuo(ind: Individuo, T: List[float], params: dict) -> Individuo:
    """Pipeline: Decodificar → MSE (§4.1) → Fitness (§4.2)."""
    ind.tensiones = decodificar_cromosoma(ind.cromosoma, params)
    
    # MSE
    n = len(T)
    ind.error = sum((t - m)**2 for t, m in zip(T, ind.tensiones)) / n
    
    # Fitness inverso
    ind.aptitud = 1.0 / (1.0 + ind.error)
    return ind

def evaluar_poblacion(poblacion: List[Individuo], T: List[float], params: dict):
    """§3.2 — Evalúa toda la población."""
    for ind in poblacion:
        evaluar_individuo(ind, T, params)


# ═══════════════════════════════════════════════════════════════════
#  4. TARGET Y MAPEO (§1.2)
# ═══════════════════════════════════════════════════════════════════

MAPEO_B = {
    't1': ('jawOpen', False),        't2': ('mouthSmileLeft', False),
    't3': ('mouthSmileRight', False),'t4': ('eyeBlinkLeft', True),
    't5': ('eyeBlinkRight', True),   't6': ('browDownLeft', True),
}

def construir_vector_target(blendshapes: Dict[str, float]) -> List[float]:
    """§1.2 — Convierte blendshapes de MediaPipe a vector T con inversiones."""
    T = []
    for key in ['t1', 't2', 't3', 't4', 't5', 't6']:
        nombre, invertir = MAPEO_B[key]
        v = max(0.0, min(1.0, blendshapes.get(nombre, 0.0)))
        T.append(1.0 - v if invertir else v)
    return T


# ═══════════════════════════════════════════════════════════════════
#  5. OPERADORES GENÉTICOS (§5, §6, §7)
# ═══════════════════════════════════════════════════════════════════

def emparejar(poblacion: List[Individuo], k: int = 3) -> List[Tuple[Individuo, Individuo]]:
    """§5 — Genera N/2 parejas por torneo k."""
    def torneo(pop, k_size):
        return max(random.sample(pop, min(k_size, len(pop))), key=lambda ind: ind.aptitud)
    return [(torneo(poblacion, k), torneo(poblacion, k)) for _ in range(len(poblacion) // 2)]

def cruza_un_punto(p1: Individuo, p2: Individuo, pc: float = 0.8) -> Tuple[Individuo, Individuo]:
    """§6.1 — Cruza de 1 punto."""
    if random.random() >= pc: return Individuo(cromosoma=p1.cromosoma), Individuo(cromosoma=p2.cromosoma)
    c = random.randint(1, len(p1.cromosoma) - 1)
    return (Individuo(cromosoma=p1.cromosoma[:c] + p2.cromosoma[c:]),
            Individuo(cromosoma=p2.cromosoma[:c] + p1.cromosoma[c:]))

def cruza_dos_puntos(p1: Individuo, p2: Individuo, pc: float = 0.8) -> Tuple[Individuo, Individuo]:
    """§6.2 — Cruza de 2 puntos."""
    if random.random() >= pc: return Individuo(cromosoma=p1.cromosoma), Individuo(cromosoma=p2.cromosoma)
    L = len(p1.cromosoma)
    c1 = random.randint(1, L - 2)
    c2 = random.randint(c1 + 1, L - 1)
    h1 = p1.cromosoma[:c1] + p2.cromosoma[c1:c2] + p1.cromosoma[c2:]
    h2 = p2.cromosoma[:c1] + p1.cromosoma[c1:c2] + p2.cromosoma[c2:]
    return Individuo(cromosoma=h1), Individuo(cromosoma=h2)

def mutar(ind: Individuo, pm: float = 0.01) -> Individuo:
    """§7.1 — Mutación Bit-flip."""
    bits = ['1' if b == '0' else '0' if random.random() < pm else b for b in ind.cromosoma]
    return Individuo(cromosoma=''.join(bits))


# ═══════════════════════════════════════════════════════════════════
#  6. GESTIÓN POBLACIONAL (§8, §9)
# ═══════════════════════════════════════════════════════════════════

def reemplazo_elitista(poblacion: List[Individuo], hijos: List[Individuo], e: int = 2) -> List[Individuo]:
    """§8 — Mantiene Top-e actuales, une con hijos y trunca a N."""
    elites = sorted(poblacion, key=lambda i: i.aptitud, reverse=True)[:e]
    return sorted(elites + hijos, key=lambda i: i.aptitud, reverse=True)[:len(poblacion)]

def podar(poblacion: List[Individuo], N: int, f_min: float, params: dict) -> List[Individuo]:
    """§9 — Filtra individuos malos y repuebla con aleatorios (diversidad)."""
    if f_min > 0.0:
        poblacion = [ind for ind in poblacion if ind.aptitud >= f_min]
    pop_ordenada = sorted(poblacion, key=lambda i: i.aptitud, reverse=True)
    if len(pop_ordenada) >= N:
        return pop_ordenada[:N]
    
    # Inyección de diversidad si faltan
    faltantes = N - len(pop_ordenada)
    res = list(pop_ordenada)
    for _ in range(faltantes):
        res.append(Individuo(cromosoma=generar_cromosoma_aleatorio(params['longitud_cromosoma'])))
    return res


# ═══════════════════════════════════════════════════════════════════
#  7. DIAGNÓSTICO Y SALIDA (§11)
# ═══════════════════════════════════════════════════════════════════

def calcular_estabilidad(M_act: List[float], M_ant: List[float]) -> float:
    """§11.1 — S: Distancia cuadrada entre fotogramas."""
    return sum((a - b)**2 for a, b in zip(M_act, M_ant)) / len(M_act)

def calcular_diversidad(poblacion: List[Individuo], L: int = 60) -> float:
    """§11.2 — D: Distancia de Hamming promedio normalizada."""
    N = len(poblacion)
    if N < 2: return 0.0
    s_ham, pares = 0.0, 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            s_ham += sum(c1 != c2 for c1, c2 in zip(poblacion[i].cromosoma, poblacion[j].cromosoma))
            pares += 1
    return (s_ham / pares) / L

def imprimir_generacion(gen: int, mejor: Individuo, promedio_aptitud: float, diversidad: float) -> None:
    print(f"  Gen {gen:>3} | Mejor F={mejor.aptitud:.6f} | E={mejor.error:.6f} | F_prom={promedio_aptitud:.6f} | D={diversidad:.4f}")

def imprimir_mejor_individuo(mejor: Individuo) -> None:
    print("\n" + "=" * 65)
    print("  MEJOR INDIVIDUO ENCONTRADO")
    print("=" * 65)
    print(f"  Cromosoma : {mejor.cromosoma}")
    print(f"  Tensiones : ", end="")
    nombres = ['m₁(Boca)', 'm₂(CigI)', 'm₃(CigD)', 'm₄(OjI)', 'm₅(OjD)', 'm₆(Cejas)']
    for nombre, valor in zip(nombres, mejor.tensiones):
        print(f"{nombre}={valor:.4f}  ", end="")
    print(f"\n  Error MSE : {mejor.error:.8f}")
    print(f"  Aptitud F : {mejor.aptitud:.8f}")
    print("=" * 65)

def graficar_comparacion(nombre: str, T: List[float], M: List[float]) -> None:
    """Genera una gráfica de barras comparando el Target vs el Resultado (M)."""
    nombres = ['Boca', 'CigI', 'CigD', 'OjI', 'OjD', 'Cejas']
    x = np.arange(len(nombres))
    width = 0.35
    
    # Estilo básico para visibilidad clara
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.bar(x - width/2, T, width, label='Target Humano (T)', color='royalblue', edgecolor='black')
    ax.bar(x + width/2, M, width, label='Gemelo Digital (M)', color='tomato', edgecolor='black')
    
    ax.set_ylabel('Tensión [0, 1]')
    ax.set_title(f'Precisión de Imitación: Expresión "{nombre.upper()}"', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_dir = '/home/luis/Documents/00IA/221189/AG01/temp'
    os.makedirs(out_dir, exist_ok=True)
        
    ruta = os.path.join(out_dir, f'comparacion_{nombre}.png')
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"\n  📊 Gráfica comparativa guardada en: {ruta}")

def graficar_evolucion(nombre: str, hist_mejor: List[float], hist_promedio: List[float]) -> None:
    """Genera gráfica de evolución de aptitud a lo largo de generaciones."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    generaciones = range(1, len(hist_mejor) + 1)
    
    ax.plot(generaciones, hist_mejor, 'forestgreen', linewidth=2, marker='o', label='Mejor aptitud (F_max)', markersize=4)
    ax.plot(generaciones, hist_promedio, 'royalblue', linewidth=2, linestyle='--', marker='s', label='Aptitud promedio (F_prom)', markersize=4, alpha=0.7)
    
    ax.set_xlabel('Generación')
    ax.set_ylabel('Aptitud F')
    ax.set_title(f'Convergencia Evolutiva: Expresión "{nombre.upper()}"', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    out_dir = '/home/luis/Documents/00IA/221189/AG01/temp'
    os.makedirs(out_dir, exist_ok=True)
    ruta = os.path.join(out_dir, f'evolucion_{nombre}.png')
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  📈 Gráfica de evolución guardada en: {ruta}")


# ═══════════════════════════════════════════════════════════════════
#  8. ORQUESTADOR EVOLUTIVO (§10, §13)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ResultadoFotograma:
    mejor: Individuo = None
    poblacion: List[Individuo] = field(default_factory=list)
    generaciones_usadas: int = 0
    razon_termino: str = ""
    historial_mejor_F: List[float] = field(default_factory=list)
    historial_promedio_F: List[float] = field(default_factory=list)

def evolucionar_fotograma(poblacion: List[Individuo], T: List[float], 
                           params: dict, config: ConfigAG) -> ResultadoFotograma:
    """§13.1 — Un ciclo completo (Fotograma)."""
    res = ResultadoFotograma()
    
    evaluar_poblacion(poblacion, T, params)
    func_cruza = cruza_un_punto if config.tipo_cruza == '1punto' else cruza_dos_puntos

    for g in range(config.G_max):
        mejor = max(poblacion, key=lambda i: i.aptitud)
        promedio = sum(i.aptitud for i in poblacion) / len(poblacion)
        diversidad = calcular_diversidad(poblacion, params['longitud_cromosoma'])
        
        res.historial_mejor_F.append(mejor.aptitud)
        res.historial_promedio_F.append(promedio)
        
        imprimir_generacion(g + 1, mejor, promedio, diversidad)
        
        # Crit 2: Error
        if mejor.error < config.epsilon:
            res.razon_termino = f"E={mejor.error:.6f} < ε"
            res.mejor, res.generaciones_usadas, res.poblacion = mejor, g + 1, poblacion
            return res
            
        # Crit 3: Estancamiento
        if g >= config.w and abs(res.historial_mejor_F[-1] - res.historial_mejor_F[-1-config.w]) < config.sigma:
            res.razon_termino = f"Estancamiento ΔF < σ"
            res.mejor, res.generaciones_usadas, res.poblacion = mejor, g + 1, poblacion
            return res

        # Flujo GA
        parejas = emparejar(poblacion, config.k_torneo)
        hijos = []
        for p1, p2 in parejas:
            h1, h2 = func_cruza(p1, p2, config.pc)
            hijos.extend([h1, h2])
            
        hijos = [mutar(h, config.pm) for h in hijos]
        evaluar_poblacion(hijos, T, params)
        poblacion = decimar_poblacion(poblacion, hijos, config, params, T)

    res.razon_termino = f"G_max={config.G_max}"
    res.mejor = max(poblacion, key=lambda i: i.aptitud)
    res.generaciones_usadas, res.poblacion = config.G_max, poblacion
    return res

def decimar_poblacion(poblacion, hijos, config, params, T):
    """Auxiliar para ordenar procesos de reemplazo."""
    pop = reemplazo_elitista(poblacion, hijos, config.elites)
    pop = podar(pop, config.N, config.f_min, params)
    for ind in pop:
        if ind.aptitud == 0.0: evaluar_individuo(ind, T, params)
    return pop


# ═══════════════════════════════════════════════════════════════════
#  9. MAIN (EXPERIMENTOS)
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  MONOLITO DE PRUEBA: ALGORITMO GENÉTICO (EVA)")
    print("═" * 60)
    
    random.seed(42)
    params = calcular_params_codificacion()
    config = ConfigAG()
    
    print(f"\n  Configuración: L={params['longitud_cromosoma']} bits, "
          f"N={config.N}, pc={config.pc}, pm={config.pm}")

    EXPRESIONES = {
        'sonrisa': [0.30, 0.85, 0.85, 0.80, 0.80, 0.70],
        'sorpresa':[0.90, 0.10, 0.10, 0.95, 0.95, 0.95],
        'tristeza':[0.05, 0.02, 0.02, 0.70, 0.70, 0.30],
    }

    # Acumuladores para validación cruzada
    X_datos = []
    Y_datos = []
    nombres_expresiones = list(EXPRESIONES.keys())

    # Pruebas de convergencia secuencial
    for nombre, T in EXPRESIONES.items():
        print(f"\n{'#' * 65}")
        print(f"  EXPRESIÓN: {nombre.upper()}")
        print(f"  Target T = {[f'{t:.2f}' for t in T]}")
        print(f"{'#' * 65}\n")

        # Generar nueva población para cada prueba (evita sesgo de convergencia previa)
        poblacion = generar_poblacion(config.N, params)
        res = evolucionar_fotograma(poblacion, T, params, config)
        
        print(f"\n  ── Razón de término: {res.razon_termino}")
        print(f"  ── Generaciones usadas: {res.generaciones_usadas}")
        
        imprimir_mejor_individuo(res.mejor)
        
        # Graficar y guardar imagen en /temp/
        graficar_comparacion(nombre, T, res.mejor.tensiones)
        graficar_evolucion(nombre, res.historial_mejor_F, res.historial_promedio_F)
        
        # Acumular datos para validación cruzada: (input=T, output=M)
        X_datos.append(T)
        Y_datos.append(res.mejor.tensiones)

    print("-" * 60)
    
    # ════════════════════════════════════════════════════════════
    #  FASE 2: VALIDACIÓN CRUZADA CON TENSORFLOW
    # ════════════════════════════════════════════════════════════    
    #
    # El modelo de red neuronal aprende a mapear T → M directamente,
    # emulando lo que el AG hace durante el ciclo evolutivo.
    # Usando K-Fold, evaluamos qué tan bien generaliza ese aprendizaje.
    
    print("\n" + "═" * 60)
    print("  VALIDACIÓN CRUZADA CON TENSORFLOW")
    print("  Red Neuronal Sustituta: T (blendshapes) → M (tensiones)")
    print("═" * 60)
    ejecutar_validacion_cruzada(X_datos, Y_datos, nombres_expresiones, k_folds=len(X_datos))

def construir_modelo_tf(n_in: int = 6, n_out: int = 6) -> tf.keras.Model:
    """Red neuronal densa para aprender el mapeo T→M."""
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(n_in,), name='capa_entrada'),
        tf.keras.layers.Dense(128, activation='relu', name='capa_oculta_1'),
        tf.keras.layers.Dense(64, activation='relu', name='capa_oculta_2'),
        tf.keras.layers.Dense(n_out, activation='sigmoid', name='capa_salida'),
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return modelo


def ejecutar_validacion_cruzada(X: list, Y: list, nombres: list, k_folds: int = 3):
    """
    K-Fold Cross-Validation usando TensorFlow.
    
    Divide el conjunto de expresiones en k particiones.
    Entrena con k-1 y valida con 1, repitiendo para cada fold.
    Genera gráficas de MSE por fold, curvas de aprendizaje y
    comparación predicción vs real para cada fold.
    """
    X_arr = np.array(X, dtype=np.float32)
    Y_arr = np.array(Y, dtype=np.float32)
    n = len(X_arr)
    
    if n < 2:
        print("  ⚠ Se necesitan al menos 2 muestras para validación cruzada.")
        return

    # Para datos pequeños usamos Leave-One-Out (k = n)
    k_folds = n if k_folds >= n else k_folds
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f"  Configuración: {k_folds}-Fold CV | Muestras: {n} | Epochs: 300")
    print(f"  Arquitectura: Dense(6→64→128→64→6) | Activación: ReLU + Sigmoid\n")
    
    historial_mse = []
    historial_mae = []
    historiales_entrenamiento = []
    predicciones_por_fold = []  # (y_real, y_pred, nombre_expresion_test)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_arr)):
        nombre_test = nombres[test_idx[0]]
        print(f"  Fold {fold_idx+1}/{k_folds} | Test: '{nombre_test}' | "
              f"Train: {[nombres[i] for i in train_idx]}")
        
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        Y_train, Y_test = Y_arr[train_idx], Y_arr[test_idx]
        
        modelo = construir_modelo_tf()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=300,
            batch_size=1,
            validation_data=(X_test, Y_test),
            verbose=0
        )
        
        # Evaluar en test
        mse_final, mae_final = modelo.evaluate(X_test, Y_test, verbose=0)
        historial_mse.append(mse_final)
        historial_mae.append(mae_final)
        historiales_entrenamiento.append(historia)
        
        # Guardar predicción para graficar
        y_pred = modelo.predict(X_test, verbose=0)[0]
        predicciones_por_fold.append((Y_test[0], y_pred, nombre_test))
        
        print(f"    MSE={mse_final:.6f}  MAE={mae_final:.6f}")
    
    print(f"\n  MSE Promedio: {np.mean(historial_mse):.6f} ± {np.std(historial_mse):.6f}")
    print(f"  MAE Promedio: {np.mean(historial_mae):.6f} ± {np.std(historial_mae):.6f}")
    
    # ── Gráfica 1: MSE y MAE por fold ──────────────────────────────────────
    graficar_metricas_cv(historial_mse, historial_mae, nombres)
    
    # ── Gráfica 2: Curvas de aprendizaje por fold ──────────────────────
    graficar_curvas_aprendizaje(historiales_entrenamiento, nombres)
    
    # ── Gráfica 3: Predicción vs Real por fold ─────────────────────
    graficar_prediccion_vs_real(predicciones_por_fold)


def graficar_metricas_cv(mse_vals: list, mae_vals: list, nombres: list) -> None:
    """Gráfica de barras: MSE y MAE por fold de validación cruzada."""
    n = len(mse_vals)
    x = np.arange(n)
    width = 0.35
    labels = [f"Fold {i+1}\n(test: {nombres[i]})" for i in range(n)]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_mse = ax.bar(x - width/2, mse_vals, width, label='MSE Test', color='tomato', edgecolor='black')
    bars_mae = ax.bar(x + width/2, mae_vals, width, label='MAE Test', color='royalblue', edgecolor='black')
    
    ax.set_title('Error por Fold — Validación Cruzada K-Fold (TF)', fontweight='bold')
    ax.set_ylabel('Error')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.axhline(np.mean(mse_vals), color='tomato', linestyle=':', linewidth=1.5, label=f'MSE prom={np.mean(mse_vals):.4f}')
    ax.axhline(np.mean(mae_vals), color='royalblue', linestyle=':', linewidth=1.5, label=f'MAE prom={np.mean(mae_vals):.4f}')
    ax.legend()
    
    ruta = '/home/luis/Documents/00IA/221189/AG01/temp/cv_metricas_por_fold.png'
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"\n  📊 [CV] Métricas por fold guardadas en: {ruta}")


def graficar_curvas_aprendizaje(historiales: list, nombres: list) -> None:
    """Curvas de pérdida train/val por epoch para cada fold."""
    n = len(historiales)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1: axes = [axes]
    
    for i, (hist, nombre, ax) in enumerate(zip(historiales, nombres, axes)):
        epochs = range(1, len(hist.history['loss']) + 1)
        ax.plot(epochs, hist.history['loss'], color='tomato', label='Train loss', linewidth=2)
        if 'val_loss' in hist.history:
            ax.plot(epochs, hist.history['val_loss'], color='royalblue', linestyle='--', label='Val loss', linewidth=2)
        ax.set_title(f"Fold {i+1} | test: '{nombre}'")
        ax.set_xlabel('Epoch')
        if i == 0: ax.set_ylabel('MSE (Loss)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle('Curvas de Aprendizaje por Fold — TensorFlow', fontweight='bold')
    ruta = '/home/luis/Documents/00IA/221189/AG01/temp/cv_curvas_aprendizaje.png'
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  📈 [CV] Curvas de aprendizaje guardadas en: {ruta}")


def graficar_prediccion_vs_real(predicciones: list) -> None:
    """Para cada fold: barras de tensión real (M_AG) vs predicción NN."""
    actuadores = ['Boca', 'CigI', 'CigD', 'OjI', 'OjD', 'Cejas']
    x = np.arange(len(actuadores))
    width = 0.35
    n = len(predicciones)
    
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1: axes = [axes]
    
    for i, (y_real, y_pred, nombre, ax) in enumerate(zip(
            [p[0] for p in predicciones],
            [p[1] for p in predicciones],
            [p[2] for p in predicciones],
            axes)):
        ax.bar(x - width/2, y_real, width, label='M real (AG)', color='forestgreen', edgecolor='black')
        ax.bar(x + width/2, y_pred, width, label='M pred (NN)', color='orange', edgecolor='black')
        ax.set_title(f"Fold {i+1}: '{nombre}'")
        ax.set_xticks(x)
        ax.set_xticklabels(actuadores)
        ax.set_ylim(0, 1.2)
        if i == 0: ax.set_ylabel('Tensión [0, 1]')
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    fig.suptitle('Predicción NN vs Real AG por Fold — Validación Cruzada', fontweight='bold')
    ruta = '/home/luis/Documents/00IA/221189/AG01/temp/cv_prediccion_vs_real.png'
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  🤖 [CV] Predicción vs Real guardado en: {ruta}")


if __name__ == "__main__":
    main()