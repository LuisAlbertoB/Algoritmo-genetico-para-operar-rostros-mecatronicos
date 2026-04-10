"""
db.py
=====
§7.1 — Base de Conocimiento y Registro de Sesiones.

Gestiona la base de datos SQLite del sistema EVA con dos capas:
    - Tablas Estáticas (§7.1): Conocimiento inmutable del modelo robótico.
    - Tablas Dinámicas (§9.2): Registro de sesiones en tiempo real.

Archivo: main/data/eva_conocimiento.db
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import asdict


# ─── Ruta por defecto de la base de datos ────────────────────────
_MODULO_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(_MODULO_DIR)), 'data')
RUTA_BD_DEFAULT = os.path.join(_DATA_DIR, 'eva_conocimiento.db')


class BaseConocimiento:
    """
    §7.1 — Gestor de la Base de Conocimiento SQLite.

    Uso:
        bc = BaseConocimiento()
        bc.conectar()
        bc.crear_schema()
        bc.poblar_conocimiento()
        config = bc.cargar_config_ag()
    """

    def __init__(self, ruta_db: str = None):
        self.ruta_db = ruta_db or RUTA_BD_DEFAULT
        self.conn = None

    def conectar(self):
        """Abre la conexión a SQLite. Crea el directorio si no existe."""
        os.makedirs(os.path.dirname(self.ruta_db), exist_ok=True)
        self.conn = sqlite3.connect(self.ruta_db)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        print(f"  🗃️ BD conectada: {self.ruta_db}")

    def cerrar(self):
        """Cierra la conexión."""
        if self.conn:
            self.conn.close()
            self.conn = None

    # ═══════════════════════════════════════════════════════════════
    #  SCHEMA — CREACIÓN DE TABLAS
    # ═══════════════════════════════════════════════════════════════

    def crear_schema(self):
        """Crea las 8 tablas (5 estáticas + 3 dinámicas) si no existen."""
        c = self.conn.cursor()

        # ── Tablas Estáticas (§7.1) ──────────────────────────────

        c.execute("""
            CREATE TABLE IF NOT EXISTS robot_modelo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                rol TEXT NOT NULL,
                rango_min REAL NOT NULL DEFAULT 0.0,
                rango_max REAL NOT NULL DEFAULT 1.0,
                orden INTEGER NOT NULL UNIQUE
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS mapeo_cinematico (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actuador_id INTEGER NOT NULL,
                rasgo_visual TEXT NOT NULL,
                factor_escala REAL NOT NULL DEFAULT 1.0,
                offset_px INTEGER NOT NULL DEFAULT 0,
                descripcion TEXT,
                FOREIGN KEY (actuador_id) REFERENCES robot_modelo(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS expresiones_ekman (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre_emocion TEXT NOT NULL UNIQUE,
                m1 REAL NOT NULL, m2 REAL NOT NULL, m3 REAL NOT NULL,
                m4 REAL NOT NULL, m5 REAL NOT NULL, m6 REAL NOT NULL,
                descripcion TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS parametros_material (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL UNIQUE,
                valor REAL NOT NULL,
                unidad TEXT NOT NULL DEFAULT 'adimensional',
                descripcion TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS config_ag (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clave TEXT NOT NULL UNIQUE,
                valor TEXT NOT NULL,
                tipo TEXT NOT NULL DEFAULT 'float',
                descripcion TEXT
            )
        """)

        # ── Tablas Dinámicas (§9.2) ──────────────────────────────

        c.execute("""
            CREATE TABLE IF NOT EXISTS sesiones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inicio TEXT NOT NULL,
                fin TEXT,
                frames_total INTEGER DEFAULT 0,
                fps_promedio REAL DEFAULT 0.0,
                config_json TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS fotogramas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sesion_id INTEGER NOT NULL,
                numero INTEGER NOT NULL,
                mejor_F REAL, mejor_E REAL,
                F_promedio REAL, diversidad REAL,
                t1 REAL, t2 REAL, t3 REAL, t4 REAL, t5 REAL, t6 REAL,
                m1 REAL, m2 REAL, m3 REAL, m4 REAL, m5 REAL, m6 REAL,
                FOREIGN KEY (sesion_id) REFERENCES sesiones(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS mejores_individuos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sesion_id INTEGER NOT NULL,
                posicion INTEGER NOT NULL,
                cromosoma TEXT,
                m1 REAL, m2 REAL, m3 REAL, m4 REAL, m5 REAL, m6 REAL,
                aptitud REAL, error REAL,
                generacion INTEGER, fotograma INTEGER,
                FOREIGN KEY (sesion_id) REFERENCES sesiones(id)
            )
        """)

        self.conn.commit()
        print("  🗃️ Schema creado (8 tablas).")

    # ═══════════════════════════════════════════════════════════════
    #  DATOS ESTÁTICOS — BASE DE CONOCIMIENTO
    # ═══════════════════════════════════════════════════════════════

    def poblar_conocimiento(self):
        """
        Inserta los datos estáticos de la Base de Conocimiento.
        Solo inserta si las tablas están vacías (idempotente).
        """
        c = self.conn.cursor()

        # ── robot_modelo ─────────────────────────────────────────
        if c.execute("SELECT COUNT(*) FROM robot_modelo").fetchone()[0] == 0:
            actuadores = [
                (1, 'm₁ Mandibular', 'Apertura vertical de la mandíbula inferior', 0.0, 1.0),
                (2, 'm₂ Cigomático Izq.', 'Elevación de la comisura labial izquierda', 0.0, 1.0),
                (3, 'm₃ Cigomático Der.', 'Elevación de la comisura labial derecha', 0.0, 1.0),
                (4, 'm₄ Orbicular Izq.', 'Apertura del párpado izquierdo', 0.0, 1.0),
                (5, 'm₅ Orbicular Der.', 'Apertura del párpado derecho', 0.0, 1.0),
                (6, 'm₆ Frontal', 'Elevación/fruncimiento de cejas', 0.0, 1.0),
            ]
            c.executemany(
                "INSERT INTO robot_modelo (orden, nombre, rol, rango_min, rango_max) VALUES (?,?,?,?,?)",
                actuadores
            )
            print("  🗃️ robot_modelo: 6 actuadores cargados.")

        # ── mapeo_cinematico ─────────────────────────────────────
        if c.execute("SELECT COUNT(*) FROM mapeo_cinematico").fetchone()[0] == 0:
            mapeos = [
                (1, 'apertura_boca', 30.0, 65, 'Desplazamiento vertical de labio inferior (px)'),
                (2, 'comisura_izquierda', 15.0, 0, 'Elevación de comisura izq. → curva de sonrisa (px)'),
                (3, 'comisura_derecha', 15.0, 0, 'Elevación de comisura der. → curva de sonrisa (px)'),
                (4, 'parpado_izquierdo', 18.0, -20, 'Altura del ojo izq. según apertura (px)'),
                (5, 'parpado_derecho', 18.0, -20, 'Altura del ojo der. según apertura (px)'),
                (6, 'cejas', 20.0, -55, 'Desplazamiento vertical de cejas (px)'),
            ]
            c.executemany(
                "INSERT INTO mapeo_cinematico (actuador_id, rasgo_visual, factor_escala, offset_px, descripcion) VALUES (?,?,?,?,?)",
                mapeos
            )
            print("  🗃️ mapeo_cinematico: 6 reglas cargadas.")

        # ── expresiones_ekman ────────────────────────────────────
        if c.execute("SELECT COUNT(*) FROM expresiones_ekman").fetchone()[0] == 0:
            ekman = [
                ('Alegría',   0.30, 0.90, 0.90, 0.80, 0.80, 0.60, 'Sonrisa amplia, ojos entrecerrados'),
                ('Tristeza',  0.10, 0.10, 0.10, 0.60, 0.60, 0.30, 'Comisuras caídas, cejas bajas'),
                ('Sorpresa',  0.80, 0.40, 0.40, 1.00, 1.00, 1.00, 'Boca abierta, ojos y cejas muy altos'),
                ('Miedo',     0.60, 0.20, 0.20, 0.90, 0.90, 0.90, 'Boca entreabierta, ojos muy abiertos'),
                ('Enojo',     0.40, 0.10, 0.10, 0.70, 0.70, 0.00, 'Mandíbula tensa, cejas fruncidas'),
                ('Asco',      0.50, 0.00, 0.00, 0.50, 0.50, 0.20, 'Boca cerrada, labios apretados'),
            ]
            c.executemany(
                "INSERT INTO expresiones_ekman (nombre_emocion, m1,m2,m3,m4,m5,m6, descripcion) VALUES (?,?,?,?,?,?,?,?)",
                ekman
            )
            print("  🗃️ expresiones_ekman: 6 emociones cargadas.")

        # ── parametros_material ──────────────────────────────────
        if c.execute("SELECT COUNT(*) FROM parametros_material").fetchone()[0] == 0:
            material = [
                ('elasticidad',       0.85, 'adimensional', 'Coef. de Young normalizado (0=rígido, 1=elástico)'),
                ('limite_deformacion', 0.95, '[0,1]',       'Tensión máxima efectiva de la silicona'),
                ('amortiguamiento',   0.30, '[0,1]',        'Factor de suavizado temporal (0=instantáneo, 1=lento)'),
                ('no_linealidad',     2.00, 'exponente',    'Exponente de la curva sigmoide de respuesta'),
                ('zona_muerta',       0.02, '[0,1]',        'Tensión mínima para producir movimiento'),
                ('friccion_estatica', 0.05, '[0,1]',        'Fricción interna al iniciar el movimiento'),
                ('histeresis',        0.03, '[0,1]',        'Diferencia entre activación y desactivación'),
                ('fatiga_temporal',   0.001, '[0,1]/frame', 'Pérdida de eficiencia con uso sostenido'),
            ]
            c.executemany(
                "INSERT INTO parametros_material (nombre, valor, unidad, descripcion) VALUES (?,?,?,?)",
                material
            )
            print("  🗃️ parametros_material: 8 constantes cargadas.")

        # ── config_ag ────────────────────────────────────────────
        config = [
            ('N',          '30',     'int',   'Tamaño de población'),
            ('k_torneo',   '3',      'int',   'Tamaño del torneo de selección'),
            ('pc',         '0.8',    'float', 'Probabilidad de cruza'),
            ('pm',         '0.01',   'float', 'Probabilidad de mutación por bit'),
            ('elites',     '2',      'int',   'Número de individuos élite'),
            ('f_min',      '0.0',    'float', 'Umbral mínimo de aptitud para poda'),
            ('G_max',      '10',     'int',   'Máx. generaciones por fotograma'),
            ('epsilon',    '0.005',  'float', 'Umbral de error aceptable'),
            ('w',          '3',      'int',   'Ventana de estancamiento'),
            ('sigma',      '0.0005', 'float', 'Umbral de estancamiento'),
            ('tipo_cruza', '1punto', 'str',   'Tipo de cruzamiento: 1punto | 2puntos'),
            ('resolucion', '0.001',  'float', 'Resolución deseada (delta) para codificación'),
        ]
        c.executemany(
            "INSERT OR IGNORE INTO config_ag (clave, valor, tipo, descripcion) VALUES (?,?,?,?)",
            config
        )
        print("  🗃️ config_ag: Hiperparámetros de AG cargados/actualizados.")

        self.conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #  LECTURA DE CONOCIMIENTO ESTÁTICO
    # ═══════════════════════════════════════════════════════════════

    def cargar_config_ag(self) -> dict:
        """
        Lee los hiperparámetros del AG desde la tabla config_ag.

        Returns:
            dict con claves como 'N', 'pc', 'pm', etc., ya tipados.
        """
        c = self.conn.cursor()
        c.execute("SELECT clave, valor, tipo FROM config_ag")
        config = {}
        for row in c.fetchall():
            clave, valor, tipo = row['clave'], row['valor'], row['tipo']
            if tipo == 'int':
                config[clave] = int(valor)
            elif tipo == 'float':
                config[clave] = float(valor)
            else:
                config[clave] = valor
        return config

    def cargar_parametros_material(self) -> dict:
        """
        Lee los parámetros de simulación de material.

        Returns:
            dict {nombre: valor} con las constantes del material.
        """
        c = self.conn.cursor()
        c.execute("SELECT nombre, valor FROM parametros_material")
        return {row['nombre']: row['valor'] for row in c.fetchall()}

    def cargar_expresiones_ekman(self) -> List[dict]:
        """
        Lee los 6 vectores de expresión de referencia.

        Returns:
            Lista de dicts con 'nombre', 'm1'..'m6'.
        """
        c = self.conn.cursor()
        c.execute("SELECT nombre_emocion, m1,m2,m3,m4,m5,m6, descripcion FROM expresiones_ekman")
        return [dict(row) for row in c.fetchall()]

    def cargar_robot_modelo(self) -> List[dict]:
        """
        Lee el catálogo de actuadores.

        Returns:
            Lista de dicts con datos de cada actuador, ordenada por 'orden'.
        """
        c = self.conn.cursor()
        c.execute("SELECT * FROM robot_modelo ORDER BY orden")
        return [dict(row) for row in c.fetchall()]

    # ═══════════════════════════════════════════════════════════════
    #  REGISTRO DINÁMICO — SESIONES
    # ═══════════════════════════════════════════════════════════════

    def registrar_sesion(self, config_dict: dict = None) -> int:
        """
        Crea un nuevo registro de sesión.

        Returns:
            ID de la sesión creada.
        """
        c = self.conn.cursor()
        config_json = json.dumps(config_dict) if config_dict else None
        c.execute(
            "INSERT INTO sesiones (inicio, config_json) VALUES (?, ?)",
            (datetime.now().isoformat(), config_json)
        )
        self.conn.commit()
        sesion_id = c.lastrowid
        print(f"  🗃️ Sesión #{sesion_id} iniciada.")
        return sesion_id

    def registrar_fotograma(self, sesion_id: int, datos: dict):
        """
        Registra los datos de un fotograma.

        Args:
            datos: dict con claves:
                numero, mejor_F, mejor_E, F_promedio, diversidad,
                t1..t6 (target), m1..m6 (tensiones)
        """
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO fotogramas 
            (sesion_id, numero, mejor_F, mejor_E, F_promedio, diversidad,
             t1,t2,t3,t4,t5,t6, m1,m2,m3,m4,m5,m6)
            VALUES (?,?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?,?)
        """, (
            sesion_id, datos.get('numero', 0),
            datos.get('mejor_F', 0), datos.get('mejor_E', 0),
            datos.get('F_promedio', 0), datos.get('diversidad', 0),
            *[datos.get(f't{i}', 0) for i in range(1, 7)],
            *[datos.get(f'm{i}', 0) for i in range(1, 7)],
        ))
        # No commit on each frame — batch at session close

    def registrar_mejores(self, sesion_id: int, mejores: list):
        """
        Registra el podio de los N mejores individuos de la sesión.

        Args:
            mejores: Lista de dicts con posicion, cromosoma, m1..m6,
                     aptitud, error, generacion, fotograma.
        """
        c = self.conn.cursor()
        for m in mejores:
            c.execute("""
                INSERT INTO mejores_individuos
                (sesion_id, posicion, cromosoma, m1,m2,m3,m4,m5,m6,
                 aptitud, error, generacion, fotograma)
                VALUES (?,?,?, ?,?,?,?,?,?, ?,?,?,?)
            """, (
                sesion_id, m['posicion'], m.get('cromosoma', ''),
                *[m.get(f'm{i}', 0) for i in range(1, 7)],
                m.get('aptitud', 0), m.get('error', 0),
                m.get('generacion', 0), m.get('fotograma', 0),
            ))
        self.conn.commit()

    def cerrar_sesion(self, sesion_id: int, frames_total: int,
                       fps_promedio: float):
        """Cierra una sesión con las estadísticas finales."""
        c = self.conn.cursor()
        c.execute("""
            UPDATE sesiones
            SET fin = ?, frames_total = ?, fps_promedio = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), frames_total, fps_promedio, sesion_id))
        self.conn.commit()
        print(f"  🗃️ Sesión #{sesion_id} cerrada ({frames_total} frames, {fps_promedio:.1f} FPS).")

    def commit(self):
        """Fuerza un commit de las transacciones pendientes."""
        if self.conn:
            self.conn.commit()


# ═══════════════════════════════════════════════════════════════════
#  PRUEBA STANDALONE
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bc = BaseConocimiento()
    bc.conectar()
    bc.crear_schema()
    bc.poblar_conocimiento()

    print("\n  ── Verificación ──")

    config = bc.cargar_config_ag()
    print(f"  Config AG: {config}")

    material = bc.cargar_parametros_material()
    print(f"  Material: {material}")

    ekman = bc.cargar_expresiones_ekman()
    for e in ekman:
        print(f"  Ekman: {e['nombre_emocion']} → [{e['m1']},{e['m2']},{e['m3']},{e['m4']},{e['m5']},{e['m6']}]")

    robot = bc.cargar_robot_modelo()
    for r in robot:
        print(f"  Actuador #{r['orden']}: {r['nombre']} — {r['rol']}")

    bc.cerrar()
    print("\n  ✓ Base de conocimiento verificada.\n")
