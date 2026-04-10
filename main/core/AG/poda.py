"""
poda.py
=======
Módulo §9 de los Fundamentos Matemáticos.

Implementa la poda poblacional para mantener el tamaño constante N,
eliminando individuos de baja calidad e inyectando diversidad
cuando la población cae por debajo del umbral.
"""

from typing import List
from genrarPoblacionInicial import Individuo, generar_cromosoma_aleatorio


def podar(poblacion: List[Individuo], 
          N: int, 
          f_min: float = 0.0, 
          params: dict = None) -> List[Individuo]:
    """
    §9 — Mecanismo de poda poblacional.
    
    §9.2 — Ordenar por aptitud descendente y truncar a N:
        P_{g+1} = Sort(P_{g+1}^{comb}, F, desc)[:N]
    
    §9.3 — Umbral de poda (opcional):
        P_{g+1} = {I ∈ P : F(I) ≥ F_min}
        Si la poda reduce la población por debajo de N,
        se rellenan los vacantes con individuos aleatorios
        (inyección de diversidad).
    
    Args:
        poblacion: Lista de individuos evaluados.
        N:         Tamaño deseado de la población.
        f_min:     Umbral mínimo de aptitud (default: 0.0 = desactivado).
        params:    Parámetros de codificación (necesario si se inyecta diversidad).
    
    Returns:
        Población podada de exactamente N individuos.
    """
    # Aplicar umbral mínimo de aptitud (§9.3)
    if f_min > 0.0:
        poblacion = [ind for ind in poblacion if ind.aptitud >= f_min]

    # Ordenar por aptitud descendente (§9.2)
    poblacion_ordenada = sorted(poblacion, key=lambda ind: ind.aptitud, reverse=True)

    # Truncar a N si sobra
    if len(poblacion_ordenada) >= N:
        return poblacion_ordenada[:N]

    # §9.3 — Si faltan individuos, inyectar diversidad con cromosomas aleatorios
    resultado = list(poblacion_ordenada)
    faltantes = N - len(resultado)

    if params is not None:
        L = params['longitud_cromosoma']
        for _ in range(faltantes):
            cromosoma = generar_cromosoma_aleatorio(L)
            nuevo = Individuo(cromosoma=cromosoma)
            resultado.append(nuevo)

    return resultado
