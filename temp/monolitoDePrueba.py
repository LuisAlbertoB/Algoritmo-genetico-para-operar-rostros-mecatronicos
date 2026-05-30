#!/usr/bin/env python3
"""
monolitoDePrueba.py
===================
Versión adaptada para clasificación (Red Neuronal) del dataset
leaf.csv, reemplazando el dataset de tendencias de consumo anterior.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.model_selection import KFold

# ── 1. PREPROCESAMIENTO DE DATOS ──────────────────────────────────────────

def cargar_y_preprocesar_datos(ruta_csv: str):
    print("  Leyendo y preprocesando dataset (leaf.csv)...")
    with open(ruta_csv, mode='r', encoding='utf-8') as f:
        # leaf.csv no tiene encabezados, usamos reader normal
        reader = list(csv.reader(f))
        
    n_samples = len(reader)
    if n_samples == 0:
        raise ValueError("Data vacia.")

    # X: Columnas 2 a 15 (14 características)
    # Y: Columna 0 (Especie)
    # Columna 1: Número de espécimen (descartada)
    
    X_raw = []
    Y_raw = []
    
    for row in reader:
        # Características numéricas (cols 2-15)
        features = [float(val) for val in row[2:]]
        X_raw.append(features)
        # Etiqueta de especie (col 0)
        Y_raw.append(row[0])
        
    X_datos = np.array(X_raw)
    
    # ── Normalización variables numéricas [0, 1] (Min-Max) ──
    for i in range(X_datos.shape[1]):
        col_min = X_datos[:, i].min()
        col_max = X_datos[:, i].max()
        if col_max > col_min:
            X_datos[:, i] = (X_datos[:, i] - col_min) / (col_max - col_min)
            
    # ── Mapeo de Clases y One-Hot Encoding para el Target ──
    # Las especies en leaf.csv pueden ser números no correlativos
    unique_species = sorted(list(set(Y_raw)), key=int)
    target_map = {species: i for i, species in enumerate(unique_species)}
    n_classes = len(unique_species)
    
    Y_datos = np.zeros((n_samples, n_classes))
    for i, species in enumerate(Y_raw):
        Y_datos[i, target_map[species]] = 1.0
        
    return X_datos, Y_datos, unique_species


# ── 2. MODELO DE RED NEURONAL SUPERVISADO ─────────────────────────────────

def construir_modelo_tf(n_in: int = 14, n_out: int = 30) -> tf.keras.Model:
    modelo = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_in,), name='capa_entrada'),
        tf.keras.layers.Dense(64, activation='relu', name='capa_oculta_1'),
        tf.keras.layers.Dense(32, activation='relu', name='capa_oculta_2'),
        tf.keras.layers.Dense(n_out, activation='softmax', name='capa_salida'),
    ])
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return modelo


# ── 3. VALIDACIÓN CRUZADA Y GRÁFICAS ──────────────────────────────────────

def ejecutar_validacion_cruzada(X: np.ndarray, Y: np.ndarray, nombres_clases: list, k_folds: int = 5):
    n = len(X)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f"  Configuración: {k_folds}-Fold CV | Muestras: {n} | Epochs: 50")
    print(f"  Arquitectura: Dense({X.shape[1]}→64→32→{Y.shape[1]}) | Activación: ReLU + Softmax\n")
    
    reporte_folds = []
    historial_accuracy = []
    historial_loss = []
    historiales_entrenamiento = []
    
    mejor_fold_idx = 0
    mejor_acc = -1.0
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  Entrenando Fold {fold_idx+1}/{k_folds}...")
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        modelo = construir_modelo_tf(n_in=X.shape[1], n_out=Y.shape[1])
        
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=16, # Tamaño de batch reducido para dataset pequeño
            validation_data=(X_test, Y_test),
            verbose=0
        )
        
        loss_final, acc_final = modelo.evaluate(X_test, Y_test, verbose=0)
        
        error_entrenamiento = historia.history['loss'][-1]
        error_validacion = loss_final
        error_total = (error_entrenamiento + error_validacion) / 2
        
        reporte_folds.append({
            'k': fold_idx + 1,
            'n_train': len(X_train),
            'n_val': len(X_test),
            'e_train': error_entrenamiento,
            'e_val': error_validacion,
            'e_total': error_total,
            'accuracy': acc_final
        })

        historial_loss.append(loss_final)
        historial_accuracy.append(acc_final)
        historiales_entrenamiento.append(historia)
        
        if acc_final > mejor_acc:
            mejor_acc = acc_final
            mejor_fold_idx = fold_idx
            
        print(f"    Validation Loss={loss_final:.4f}  Accuracy={acc_final*100:.2f}%")

    # ── IMPRESIÓN DEL REPORTE FINAL ACADÉMICO ──
    print("\n" + "═" * 80)
    print("  RESUMEN DE VALIDACIÓN CRUZADA: CLASIFICACIÓN DE HOJAS")
    print("═" * 80)
    print(f"  Se ha aplicado la validación cruzada para K={k_folds}")
    print("-" * 80)
    print(f"{'k':<3} | {'Obs. Train':<10} | {'Obs. Val':<10} | {'Err. Train':<12} | {'Err. Val':<12} | {'Err. Total':<12}")
    print("-" * 80)
    
    for r in reporte_folds:
        print(f"{r['k']:<3} | {r['n_train']:<10} | {r['n_val']:<10} | {r['e_train']:<12.6f} | {r['e_val']:<12.6f} | {r['e_total']:<12.6f}")
    
    print("-" * 80)
    print(f"  Accuracy Final Promedio: {np.mean(historial_accuracy)*100:.2f}%")
    print(f"  El mejor modelo es el Fold #{mejor_fold_idx + 1} (Accuracy: {mejor_acc*100:.2f}%)")
    print("═" * 80)
    
    graficar_metricas_cv(historial_accuracy, historial_loss)
    graficar_evolucion_mejor_modelo(historiales_entrenamiento[mejor_fold_idx], mejor_fold_idx)


def graficar_metricas_cv(acc_vals: list, loss_vals: list) -> None:
    n = len(acc_vals)
    x = np.arange(n)
    width = 0.35
    labels = [f"Fold {i+1}" for i in range(n)]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width/2, acc_vals, width, label='Accuracy', color='forestgreen', edgecolor='black')
    ax1.set_ylabel('Accuracy', color='forestgreen')
    ax1.set_ylim(0, 1.1)
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, loss_vals, width, label='Loss (Categorical)', color='tomato', edgecolor='black')
    ax2.set_ylabel('Categorical Crossentropy Loss', color='tomato')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    plt.title('Métricas por Fold: Clasificación de Hojas (Leaf)', fontweight='bold')
    
    ruta = '/home/luis/Documents/00IA/221189/AG01/temp/cv_metricas_clasificacion.png'
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"\n  📊 [CV] Métricas generales guardadas en: {ruta}")


def graficar_evolucion_mejor_modelo(historia, fold_idx: int) -> None:
    epochs = range(1, len(historia.history['loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, historia.history['loss'], 'tomato', label='Train Loss', lw=2)
    ax1.plot(epochs, historia.history['val_loss'], 'darkred', linestyle='--', label='Val Loss', lw=2)
    ax1.set_title('Evolución del Error Entrenamiento (Loss)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Categorical Crossentropy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(epochs, historia.history['accuracy'], 'forestgreen', label='Train Accuracy', lw=2)
    ax2.plot(epochs, historia.history['val_accuracy'], 'darkgreen', linestyle='--', label='Val Accuracy', lw=2)
    ax2.set_title('Evolución de la Precisión (Accuracy)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle(f"Evolución del Mejor Modelo - Dataset Leaf (Fold {fold_idx + 1})", fontsize=14, fontweight='bold', y=1.02)
    
    ruta = '/home/luis/Documents/00IA/221189/AG01/temp/mejor_modelo_evolucion.png'
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()
    print(f"  📈 [CV] Gráfica de evolución para el mejor modelo guardada en: {ruta}")


# ── 4. MAIN ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 65)
    print("  MONOLITO CLASIFICADOR: DATASET LEAF (HOJAS)")
    print("═" * 65)
    
    ruta_dataset = '/home/luis/Documents/00IA/221189/AG01/temp/leaf.csv'
    if not os.path.exists(ruta_dataset):
        print(f"  ⚠ Dataset no encontrado en la ruta esperada:\n  {ruta_dataset}")
        return
        
    X_datos, Y_datos, nombres_clases = cargar_y_preprocesar_datos(ruta_dataset)
    
    print(f"\n  Dataset Summary")
    print(f"  {'─' * 15}")
    print(f"  El dataset consta de {len(X_datos)} observaciones.")
    print(f"  Cada observación tiene {X_datos.shape[1]} características morfológicas.")
    print(f"  Número de especies (clases): {len(nombres_clases)}")
    print(f"  Se concluye que la salida es de tipo 'una clase definida' (Clasificación de Especies).")
    
    print("\n" + "═" * 65)
    print("  FASE 2: VALIDACIÓN CRUZADA CON TENSORFLOW")
    print("═" * 65)
    ejecutar_validacion_cruzada(X_datos, Y_datos, nombres_clases, k_folds=5)


if __name__ == '__main__':
    main()