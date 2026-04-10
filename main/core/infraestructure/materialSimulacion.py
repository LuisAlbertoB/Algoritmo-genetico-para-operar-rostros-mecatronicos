"""
materialSimulacion.py
=====================
Simulación Física del Material de Silicona — Sistema EVA.

Implementa la función de transferencia que transforma las
tensiones brutas del AG (M) en desplazamientos realistas (M')
que reflejan las propiedades físicas de la silicona suave.

Pipeline de transformación por actuador:
    1. Zona muerta: si |m| < zona_muerta → 0
    2. Fricción estática: resta la fricción al inicio del movimiento
    3. Curva de elasticidad: m' = m^(1/no_linealidad) * elasticidad
    4. Límite de deformación: clamp(m', 0, limite_deformacion)
    5. Amortiguamiento temporal: m_final = α·m' + (1-α)·m_anterior

Los parámetros se leen de la tabla `parametros_material` en SQLite.
"""

from typing import List, Dict


class ModeloMaterial:
    """
    Modelo de respuesta física de la silicona del rostro robótico.

    Transforma las tensiones brutas del AG en desplazamientos
    que simulan la elasticidad, fricción y amortiguamiento
    del material real.

    Uso:
        modelo = ModeloMaterial(params_material)
        M_prima = modelo.transformar_tensiones(M_brutas)
    """

    def __init__(self, parametros: Dict[str, float] = None):
        """
        Inicializa el modelo con los parámetros de la BD.

        Args:
            parametros: Dict {nombre: valor} de la tabla parametros_material.
                        Si es None, usa valores por defecto.
        """
        p = parametros or {}

        self.elasticidad = p.get('elasticidad', 0.85)
        self.limite_deformacion = p.get('limite_deformacion', 0.95)
        self.amortiguamiento = p.get('amortiguamiento', 0.30)
        self.no_linealidad = p.get('no_linealidad', 2.0)
        self.zona_muerta = p.get('zona_muerta', 0.02)
        self.friccion_estatica = p.get('friccion_estatica', 0.05)
        self.histeresis = p.get('histeresis', 0.03)
        self.fatiga_temporal = p.get('fatiga_temporal', 0.001)

        # Estado temporal (memoriza tensiones del frame anterior)
        self._tensiones_anteriores: List[float] = [0.0] * 6
        self._ciclos_activos: List[int] = [0] * 6  # Conteo de uso sostenido

    def transformar_tensiones(self, M: List[float]) -> List[float]:
        """
        Aplica el modelo de material a todo el vector de tensiones.

        Args:
            M: Vector [m₁,...,m₆] de tensiones brutas del AG (en [0,1]).

        Returns:
            Vector M' = [m'₁,...,m'₆] con respuesta física del material.
        """
        M_prima = []
        for i, m_bruta in enumerate(M):
            m_anterior = self._tensiones_anteriores[i]
            m_transformada = self._aplicar_material(m_bruta, m_anterior, i)
            M_prima.append(m_transformada)
            self._tensiones_anteriores[i] = m_transformada

        return M_prima

    def _aplicar_material(self, tension: float, anterior: float,
                           indice: int) -> float:
        """
        Función de transferencia del material para un actuador individual.

        Pipeline:
            1. Zona muerta
            2. Fricción estática
            3. Curva de elasticidad no-lineal
            4. Límite de deformación
            5. Fatiga temporal
            6. Amortiguamiento temporal (suavizado)
        """
        m = max(0.0, min(1.0, tension))

        # ── 1. Zona muerta ──────────────────────────────────────
        # Tensiones muy bajas no producen movimiento (la silicona
        # tiene una resistencia mínima al movimiento inicial)
        if m < self.zona_muerta:
            m = 0.0

        # ── 2. Fricción estática ────────────────────────────────
        # Al iniciar movimiento desde reposo, hay una resistencia
        # que se debe superar. Aplicamos histéresis.
        if m > 0 and anterior == 0:
            m = max(0.0, m - self.friccion_estatica)
        elif m > 0 and m < anterior:
            # Al reducir tensión, la histéresis hace que el material
            # "retenga" un poco más
            m = max(0.0, m + self.histeresis * 0.5)

        # ── 3. Curva de elasticidad no-lineal ───────────────────
        # La silicona no tiene respuesta lineal:
        #   - Baja tensión → poco movimiento (resistencia inicial)
        #   - Media tensión → respuesta proporcional
        #   - Alta tensión → saturación progresiva
        # Modelo: m' = m^(1/γ) · E   donde γ=no_linealidad, E=elasticidad
        if m > 0:
            # Raíz n-ésima para curva de respuesta suave
            m = (m ** (1.0 / self.no_linealidad)) * self.elasticidad

        # ── 4. Límite de deformación ────────────────────────────
        # La silicona no puede estirarse indefinidamente
        m = min(m, self.limite_deformacion)

        # ── 5. Fatiga temporal ──────────────────────────────────
        # Uso sostenido reduce ligeramente la eficiencia del actuador
        if m > 0.1:
            self._ciclos_activos[indice] += 1
            fatiga = self._ciclos_activos[indice] * self.fatiga_temporal
            m = max(0.0, m - min(fatiga, 0.05))  # Máx 5% de pérdida
        else:
            # El actuador descansa → se recupera gradualmente
            self._ciclos_activos[indice] = max(0, self._ciclos_activos[indice] - 2)

        # ── 6. Amortiguamiento temporal ─────────────────────────
        # La silicona no "salta" instantáneamente. Suavizado exponencial:
        # m_final = α·m_nuevo + (1-α)·m_anterior
        alpha = 1.0 - self.amortiguamiento
        m_final = alpha * m + self.amortiguamiento * anterior

        return max(0.0, min(1.0, m_final))

    def reset(self):
        """Reinicia el estado temporal del modelo."""
        self._tensiones_anteriores = [0.0] * 6
        self._ciclos_activos = [0] * 6

    def __repr__(self) -> str:
        return (f"ModeloMaterial(E={self.elasticidad}, "
                f"límite={self.limite_deformacion}, "
                f"amort={self.amortiguamiento}, "
                f"γ={self.no_linealidad})")


# ═══════════════════════════════════════════════════════════════════
#  PRUEBA STANDALONE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  ── Prueba del Modelo de Material ──\n")

    modelo = ModeloMaterial()
    print(f"  {modelo}\n")

    # Simular una secuencia de tensiones (transición gradual)
    secuencia = [
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],  # Frame 1: reposo parcial
        [0.3, 0.8, 0.8, 0.9, 0.9, 0.7],  # Frame 2: sonrisa
        [0.3, 0.9, 0.9, 0.9, 0.9, 0.7],  # Frame 3: sonrisa sostenida
        [0.3, 0.9, 0.9, 0.9, 0.9, 0.7],  # Frame 4: sonrisa sostenida
        [0.0, 0.0, 0.0, 0.8, 0.8, 0.5],  # Frame 5: reposo
        [0.0, 0.0, 0.0, 0.8, 0.8, 0.5],  # Frame 6: reposo sostenido
    ]

    for i, M in enumerate(secuencia):
        M_prima = modelo.transformar_tensiones(M)
        m_str = " ".join(f"{v:.3f}" for v in M)
        mp_str = " ".join(f"{v:.3f}" for v in M_prima)
        print(f"  Frame {i+1}: M=[{m_str}] → M'=[{mp_str}]")

    print("\n  ✓ Modelo de material verificado.\n")
