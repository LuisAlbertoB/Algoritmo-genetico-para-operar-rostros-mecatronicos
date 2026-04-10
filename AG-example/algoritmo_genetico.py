#!/usr/bin/env python3
"""
Convierte variable a representación binaria.

Función de aptitud: f(x) = x * cos(3x) + sin(7x)
Límites: [a, b] = [8, 12]
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Individuo:
    """Representa un individuo en la población del AG."""
    cadena_bits: str
    x: float
    aptitud: float
    
    def __str__(self) -> str:
        return f"Bits: {self.cadena_bits} | x: {self.x:.6f} | f(x): {self.aptitud:.6f}"


def funcion_aptitud(x: float) -> float:
    """
    Función de aptitud f(x) = x * cos(3x) + sin(7x)
    
    Args:
        x: Valor en el dominio [8, 12]
    
    Returns:
        Valor de aptitud calculado
    """
    ##return x * math.cos(3 * x) + math.sin(7 * x)
    ##return 0.25
    return 0.1 * x * abs(math.cos(x))



def calcular_bits(a: float, b: float, resolucion: float = 1e-3) -> dict:
    """
    Calcula los parámetros del sistema binario para el mapeo.
    
    Args:
        a: Límite inferior del rango
        b: Límite superior del rango
        resolucion: Resolución deseada del problema (default: 10^-3)
    
    Returns:
        Diccionario con todos los parámetros calculados
    """
    # Cálculos del sistema
    rango = b - a
    puntos_problema = int(rango / resolucion) + 1
    bits_sistema = math.ceil(math.log2(puntos_problema))
    puntos_sistema = 2 ** bits_sistema
    resolucion_sistema = rango / (puntos_sistema - 1)
    
    return {
        'a': a,
        'b': b,
        'rango': rango,
        'resolucion_problema': resolucion,
        'puntos_problema': puntos_problema,
        'bits_sistema': bits_sistema,
        'puntos_sistema': puntos_sistema,
        'resolucion_sistema': resolucion_sistema
    }


def decimal_a_binario(valor_decimal: int, num_bits: int) -> str:
    """Convierte un valor decimal a cadena binaria de longitud fija."""
    return format(valor_decimal, f'0{num_bits}b')


def binario_a_decimal(cadena_bits: str) -> int:
    """Convierte una cadena binaria a valor decimal."""
    return int(cadena_bits, 2)


def mapear_a_x(valor_decimal: int, params: dict) -> float:
    """
    Mapea un valor decimal al espacio real [a, b].
    
    x = a + (valor_decimal * rango) / (2^bits - 1)
    """
    max_decimal = params['puntos_sistema'] - 1
    x = params['a'] + (valor_decimal * params['rango']) / max_decimal
    return x


def generar_poblacion(num_individuos: int, params: dict) -> List[Individuo]:
    """
    Genera una población aleatoria de individuos.
    
    Args:
        num_individuos: Cantidad de individuos a generar
        params: Parámetros del sistema binario
    
    Returns:
        Lista de individuos generados
    """
    poblacion = []
    max_decimal = params['puntos_sistema'] - 1
    
    for _ in range(num_individuos):
        # Generar valor decimal aleatorio
        valor_decimal = random.randint(0, max_decimal)
        
        # Convertir a binario
        cadena_bits = decimal_a_binario(valor_decimal, params['bits_sistema'])
        
        # Mapear a valor x
        x = mapear_a_x(valor_decimal, params)
        
        # Calcular aptitud
        aptitud = funcion_aptitud(x)
        
        # Crear individuo
        individuo = Individuo(cadena_bits=cadena_bits, x=x, aptitud=aptitud)
        poblacion.append(individuo)
    
    return poblacion


def imprimir_parametros(params: dict) -> None:
    """Imprime los parámetros del sistema en consola."""
    print("=" * 60)
    print("PARAMETROS DEL SISTEMA DE CODIFICACION BINARIA")
    print("=" * 60)
    print(f"  Limites [a, b]         : [{params['a']}, {params['b']}]")
    print(f"  Rango                  : {params['rango']}")
    print(f"  Resolucion Problema    : {params['resolucion_problema']}")
    print(f"  Puntos Problema        : {params['puntos_problema']}")
    print(f"  Bits Sistema           : {params['bits_sistema']}")
    print(f"  Puntos Sistema (2^bits): {params['puntos_sistema']}")
    print(f"  Resolucion Sistema     : {params['resolucion_sistema']:.7f}")
    print("=" * 60)


def imprimir_poblacion(poblacion: List[Individuo]) -> None:
    """Imprime la información de cada individuo en la población."""
    print("\nPOBLACION DE INDIVIDUOS")
    print("-" * 60)
    print(f"{'#':>2} | {'Cadena de Bits':^14} | {'x':^12} | {'f(x)':^12}")
    print("-" * 60)
    
    for i, ind in enumerate(poblacion, 1):
        print(f"{i:>2} | {ind.cadena_bits:^14} | {ind.x:>12.6f} | {ind.aptitud:>12.6f}")
    
    print("-" * 60)
    
    # Estadísticas
    aptitudes = [ind.aptitud for ind in poblacion]
    mejor = max(poblacion, key=lambda x: x.aptitud)
    peor = min(poblacion, key=lambda x: x.aptitud)
    
    print(f"\nESTADISTICAS:")
    print(f"  Mejor aptitud  : {max(aptitudes):.6f} (x = {mejor.x:.6f})")
    print(f"  Peor aptitud   : {min(aptitudes):.6f} (x = {peor.x:.6f})")
    print(f"  Aptitud media  : {np.mean(aptitudes):.6f}")
    print(f"  Desv. estandar : {np.std(aptitudes):.6f}")


# ========== OPERADORES GENETICOS ==========

def seleccion_torneo(poblacion: List[Individuo], k: int = 3) -> Individuo:
    """
    Selecciona un individuo mediante torneo.
    
    Args:
        poblacion: Lista de individuos
        k: Tamaño del torneo
    
    Returns:
        Mejor individuo del torneo
    """
    competidores = random.sample(poblacion, k)
    return max(competidores, key=lambda ind: ind.aptitud)


def emparejar(poblacion: List[Individuo]) -> List[Tuple[Individuo, Individuo]]:
    """
    Empareja individuos para cruza mediante selección por torneo.
    
    Args:
        poblacion: Lista de individuos
    
    Returns:
        Lista de parejas de individuos
    """
    parejas = []
    num_parejas = len(poblacion) // 2
    
    for _ in range(num_parejas):
        padre1 = seleccion_torneo(poblacion)
        padre2 = seleccion_torneo(poblacion)
        parejas.append((padre1, padre2))
    
    return parejas


def cruza_un_punto(padre1: Individuo, padre2: Individuo, params: dict) -> Tuple[Individuo, Individuo]:
    """
    Realiza cruza de un punto entre dos padres.
    
    Args:
        padre1: Primer padre
        padre2: Segundo padre
        params: Parámetros del sistema
    
    Returns:
        Dos hijos resultantes de la cruza
    """
    punto = random.randint(1, len(padre1.cadena_bits) - 1)
    
    # Crear hijos
    hijo1_bits = padre1.cadena_bits[:punto] + padre2.cadena_bits[punto:]
    hijo2_bits = padre2.cadena_bits[:punto] + padre1.cadena_bits[punto:]
    
    # Decodificar y evaluar hijo 1
    decimal1 = binario_a_decimal(hijo1_bits)
    x1 = mapear_a_x(decimal1, params)
    apt1 = funcion_aptitud(x1)
    hijo1 = Individuo(cadena_bits=hijo1_bits, x=x1, aptitud=apt1)
    
    # Decodificar y evaluar hijo 2
    decimal2 = binario_a_decimal(hijo2_bits)
    x2 = mapear_a_x(decimal2, params)
    apt2 = funcion_aptitud(x2)
    hijo2 = Individuo(cadena_bits=hijo2_bits, x=x2, aptitud=apt2)
    
    return hijo1, hijo2


def mutar(individuo: Individuo, params: dict, prob_mutacion: float = 0.01) -> Individuo:
    """
    Aplica mutación por inversión de bit.
    
    Args:
        individuo: Individuo a mutar
        params: Parámetros del sistema
        prob_mutacion: Probabilidad de mutación por bit
    
    Returns:
        Individuo mutado
    """
    bits_lista = list(individuo.cadena_bits)
    
    for i in range(len(bits_lista)):
        if random.random() < prob_mutacion:
            bits_lista[i] = '1' if bits_lista[i] == '0' else '0'
    
    nueva_cadena = ''.join(bits_lista)
    
    # Re-evaluar
    decimal = binario_a_decimal(nueva_cadena)
    x = mapear_a_x(decimal, params)
    apt = funcion_aptitud(x)
    
    return Individuo(cadena_bits=nueva_cadena, x=x, aptitud=apt)


def podar_elitismo(poblacion: List[Individuo], nueva_generacion: List[Individuo], 
                    num_elites: int = 2) -> List[Individuo]:
    """
    Combina la población actual con la nueva, preservando los mejores (elitismo).
    
    Args:
        poblacion: Población actual
        nueva_generacion: Nueva generación
        num_elites: Número de mejores individuos a preservar
    
    Returns:
        Nueva población podada
    """
    # Ordenar población actual y tomar los mejores
    poblacion_ordenada = sorted(poblacion, key=lambda ind: ind.aptitud, reverse=True)
    elites = poblacion_ordenada[:num_elites]
    
    # Combinar elites con nueva generación
    combinada = elites + nueva_generacion
    
    # Ordenar y tomar los mejores n individuos
    combinada_ordenada = sorted(combinada, key=lambda ind: ind.aptitud, reverse=True)
    
    return combinada_ordenada[:len(poblacion)]


# ========== GRAFICAS ==========

def graficar_funcion_y_poblacion(poblacion: List[Individuo], params: dict, 
                                  nombre_archivo: str = "aptitud_individuos.png") -> None:
    """
    Genera una gráfica de la función de aptitud con los individuos marcados.
    
    Args:
        poblacion: Lista de individuos
        params: Parámetros del sistema
        nombre_archivo: Nombre del archivo de salida
    """
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Generar curva de la función
    x_continuo = np.linspace(params['a'], params['b'], 1000)
    y_continuo = [funcion_aptitud(x) for x in x_continuo]
    
    # Graficar función
    ax.plot(x_continuo, y_continuo, 'b-', linewidth=2, label='f(x) = x·cos(3x) + sin(7x)', alpha=0.8)
    
    # Graficar individuos
    x_individuos = [ind.x for ind in poblacion]
    y_individuos = [ind.aptitud for ind in poblacion]
    
    ax.scatter(x_individuos, y_individuos, c='red', s=100, zorder=5, 
               edgecolors='darkred', linewidths=2, label=f'Individuos (n={len(poblacion)})')
    
    # Marcar el mejor individuo
    mejor = max(poblacion, key=lambda x: x.aptitud)
    ax.scatter([mejor.x], [mejor.aptitud], c='lime', s=200, zorder=6,
              edgecolors='darkgreen', linewidths=3, marker='*', label=f'Mejor: x={mejor.x:.4f}')
    
    # Configurar gráfica
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Algoritmo Genetico: Funcion de Aptitud e Individuos',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(params['a'] - 0.2, params['b'] + 0.2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"\nGrafica guardada")
    plt.close()


def graficar_evolucion(historial_mejor: List[float], historial_promedio: List[float],
                       nombre_archivo: str = "evolucion_aptitud.png") -> None:
    """
    Genera gráfica de evolución de aptitud a lo largo de generaciones.
    
    Args:
        historial_mejor: Lista de mejores aptitudes por generación
        historial_promedio: Lista de aptitudes promedio por generación
        nombre_archivo: Nombre del archivo de salida
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generaciones = range(1, len(historial_mejor) + 1)
    
    ax.plot(generaciones, historial_mejor, 'g-', linewidth=2, marker='o', 
            label='Mejor aptitud', markersize=4)
    ax.plot(generaciones, historial_promedio, 'b--', linewidth=2, marker='s',
            label='Aptitud promedio', markersize=4, alpha=0.7)
    
    ax.set_xlabel('Generacion', fontsize=12)
    ax.set_ylabel('Aptitud', fontsize=12)
    ax.set_title('Evolucion de Aptitud por Generacion', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    print(f"Grafica de evolucion guardada")
    plt.close()

def evolucionar(poblacion: List[Individuo], params: dict, num_generaciones: int = 50,
                prob_cruza: float = 0.8, prob_mutacion: float = 0.01) -> Tuple[Individuo, List[float], List[float]]:
    """
    Ejecuta el algoritmo genético durante un número de generaciones.
    
    Args:
        poblacion: Población inicial
        params: Parámetros del sistema
        num_generaciones: Número de generaciones a evolucionar
        prob_cruza: Probabilidad de cruza
        prob_mutacion: Probabilidad de mutación
    
    Returns:
        Mejor individuo encontrado, historial de mejores aptitudes, historial de aptitudes promedio
    """
    historial_mejor = []
    historial_promedio = []
    
    for gen in range(num_generaciones):
        # Estadísticas de la generación actual
        aptitudes = [ind.aptitud for ind in poblacion]
        mejor_actual = max(poblacion, key=lambda ind: ind.aptitud)
        
        historial_mejor.append(mejor_actual.aptitud)
        historial_promedio.append(np.mean(aptitudes))
        
        if gen % 10 == 0:
            print(f"Generacion {gen:>3} - Mejor: {mejor_actual.aptitud:>10.6f} - Promedio: {np.mean(aptitudes):>10.6f}")
        
        # Crear nueva generación
        nueva_generacion = []
        
        # Emparejamiento y cruza
        parejas = emparejar(poblacion)
        for padre1, padre2 in parejas:
            if random.random() < prob_cruza:
                hijo1, hijo2 = cruza_un_punto(padre1, padre2, params)
            else:
                hijo1, hijo2 = padre1, padre2
            
            nueva_generacion.append(hijo1)
            nueva_generacion.append(hijo2)
        
        # Mutación
        nueva_generacion = [mutar(ind, params, prob_mutacion) for ind in nueva_generacion]
        
        # Poda con elitismo
        poblacion = podar_elitismo(poblacion, nueva_generacion, num_elites=2)
    
    # Estadísticas finales
    mejor_final = max(poblacion, key=lambda ind: ind.aptitud)
    return mejor_final, historial_mejor, historial_promedio


def imprimir_reporte_mejor(mejor: Individuo, params: dict) -> None:
    """
    Imprime un reporte detallado del mejor individuo.
    
    Args:
        mejor: Mejor individuo encontrado
        params: Parámetros del sistema
    """
    print("\n" + "=" * 60)
    print("REPORTE DEL MEJOR INDIVIDUO")
    print("=" * 60)
    print(f"  Cadena de bits : {mejor.cadena_bits}")
    print(f"  Valor decimal  : {binario_a_decimal(mejor.cadena_bits)}")
    print(f"  Valor x        : {mejor.x:.8f}")
    print(f"  Aptitud f(x)   : {mejor.aptitud:.8f}")
    print("=" * 60)


def main():
    """Función principal del algoritmo genético."""
    print("\n" + "#" * 60)
    print("  ALGORITMO GENETICO COMPLETO")
    print("#" * 60 + "\n")
    
    # Semilla para reproducibilidad (opcional: comentar para resultados aleatorios)
    random.seed(42)
    
    # Definir límites y precisión
    a, b = 8, 12
    precision = 1e-3
    
    # Parámetros del AG
    num_individuos = 10
    num_generaciones = 50
    prob_cruza = 0.8
    prob_mutacion = 0.01
    
    # Calcular parámetros del sistema
    params = calcular_bits(a, b, precision)
    imprimir_parametros(params)
    
    # Generar población inicial
    print("\n" + "-" * 60)
    print("POBLACION INICIAL")
    print("-" * 60)
    poblacion = generar_poblacion(num_individuos, params)
    imprimir_poblacion(poblacion)
    
    # Evolucionar
    print("\n" + "=" * 60)
    print("EVOLUCION")
    print("=" * 60)
    mejor, hist_mejor, hist_promedio = evolucionar(poblacion, params, num_generaciones, 
                                                    prob_cruza, prob_mutacion)
    
    # Reporte del mejor individuo
    imprimir_reporte_mejor(mejor, params)
    
    # Generar gráficas
    print("\n" + "-" * 60)
    print("GENERANDO GRAFICAS")
    print("-" * 60)
    
    # Recrear población final para graficar
    poblacion_final = generar_poblacion(num_individuos, params)
    poblacion_final[0] = mejor  # Asegurar que el mejor esté en la gráfica
    
    graficar_funcion_y_poblacion(poblacion_final, params)
    graficar_evolucion(hist_mejor, hist_promedio)
    
    print("\n" + "#" * 60)
    print("Ejecucion completada exitosamente.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
