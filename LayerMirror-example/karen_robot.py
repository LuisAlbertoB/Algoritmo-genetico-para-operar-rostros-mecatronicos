"""
karen_robot.py  — Compatible con MediaPipe >= 0.10
=====================================================
Detecta expresiones faciales con la webcam y muestra en tiempo real
una cara animada estilo "Karen" con píxeles verdes.

Requisitos:
    pip install opencv-python mediapipe pygame numpy
"""

import cv2
import pygame
import numpy as np
import random
import math
import sys

# ── Intentar importar MediaPipe con la API nueva ──────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    NEW_MP_API = True
except Exception:
    NEW_MP_API = False

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
CAM_W, CAM_H   = 640, 480
ANIM_W, ANIM_H = 480, 480
PIXEL          = 8
FPS            = 30

GREEN_BRIGHT = (0, 255, 80)
GREEN_MID    = (0, 200, 50)
GREEN_DIM    = (0, 100, 30)
CYAN         = (0, 255, 220)
SCREEN_BG    = (4, 8, 4)

# ──────────────────────────────────────────────
# ALGORITMO GENÉTICO
# ──────────────────────────────────────────────
POP_SIZE       = 30
MUTATION_RATE  = 0.15
CROSSOVER_RATE = 0.7

def rand_ind():
    return [random.random() for _ in range(6)]

def crossover(a, b):
    if random.random() < CROSSOVER_RATE:
        pt = random.randint(1, 5)
        return a[:pt] + b[pt:]
    return a[:]

def mutate(ind):
    return [min(1.0, max(0.0, g + random.gauss(0, 0.1)))
            if random.random() < MUTATION_RATE else g
            for g in ind]

def fitness(ind, target):
    d = sum((a - b)**2 for a, b in zip(ind, target))
    return 1.0 / (1.0 + d)

def evolve(pop, target):
    scored = sorted(pop, key=lambda i: fitness(i, target), reverse=True)
    elite  = scored[:5]
    new_pop = elite[:]
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(elite, 2)
        new_pop.append(mutate(crossover(p1, p2)))
    return new_pop

# ──────────────────────────────────────────────
# DETECTOR HAAR (fallback sin MediaPipe)
# ──────────────────────────────────────────────
class FaceDetectorHaar:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")
        self._prev = [0.3, 0.7, 0.8, 0.8, 0.5, 0.9]
        self._tick = 0

    def detect(self, frame):
        self._tick += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))

        if len(faces) == 0:
            return self._prev, frame

        x, y, w, h = faces[0]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,80), 2)

        roi = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 3)

        eye_open   = 0.9 if len(eyes) >= 2 else (0.6 if len(eyes)==1 else 0.4)
        smile      = 0.5 + 0.3 * math.sin(self._tick * 0.03)
        mouth_open = 0.2 + 0.15 * abs(math.sin(self._tick * 0.05))
        brow       = 0.5
        brightness = 0.85

        genes = [mouth_open, smile, eye_open, eye_open, brow, brightness]
        self._prev = genes
        return genes, frame

# ──────────────────────────────────────────────
# DETECTOR MEDIAPIPE (API >= 0.10)
# ──────────────────────────────────────────────
class FaceDetectorMediaPipe:
    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._prev = [0.3, 0.7, 0.8, 0.8, 0.5, 0.9]

    def _blend(self, shapes, name):
        for s in shapes:
            if s.category_name == name:
                return s.score
        return 0.0

    def detect(self, frame):
        import mediapipe as mp
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        if not result.face_landmarks:
            return self._prev, frame

        lms  = result.face_landmarks[0]
        fh, fw = frame.shape[:2]
        for lm in lms[::5]:
            cv2.circle(frame, (int(lm.x*fw), int(lm.y*fh)), 1, (0,255,80), -1)

        bs = result.face_blendshapes[0] if result.face_blendshapes else []

        mouth_open = self._blend(bs, "jawOpen")
        smile      = 0.5 + 0.5 * max(self._blend(bs, "mouthSmileLeft"),
                                      self._blend(bs, "mouthSmileRight"))
        eye_l      = 1.0 - self._blend(bs, "eyeBlinkLeft")
        eye_r      = 1.0 - self._blend(bs, "eyeBlinkRight")
        brow       = 1.0 - max(self._blend(bs, "browDownLeft"),
                                self._blend(bs, "browDownRight"))
        brightness = 0.7 + 0.3 * mouth_open

        genes = [mouth_open, smile, eye_l, eye_r, brow, brightness]
        self._prev = genes
        return genes, frame

# ──────────────────────────────────────────────
# RENDERIZADO KAREN PIXEL-ART
# ──────────────────────────────────────────────
def px(surf, gx, gy, color):
    pygame.draw.rect(surf, color, (gx*PIXEL, gy*PIXEL, PIXEL-1, PIXEL-1))

def draw_karen(surf, genes, tick, font_small):
    surf.fill(SCREEN_BG)
    GW = ANIM_W // PIXEL
    GH = ANIM_H // PIXEL
    cx = GW // 2
    cy = GH // 2
    br = genes[5]

    def gc(base=GREEN_BRIGHT):
        return (int(base[0]*br), int(base[1]*br), int(base[2]*br))

    # Contorno cara
    fw, fh = 38, 44
    fx, fy = cx - fw//2, cy - fh//2
    for dx in range(fw):
        for dy in range(fh):
            edge   = dx==0 or dx==fw-1 or dy==0 or dy==fh-1
            corner = ((dx<2 and dy<2) or (dx<2 and dy>fh-3) or
                      (dx>fw-3 and dy<2) or (dx>fw-3 and dy>fh-3))
            if edge and not corner:
                px(surf, fx+dx, fy+dy, gc(GREEN_MID))

    # Ojos
    def draw_eye(ex, ey, openness):
        ew = 8
        max_eh = 6
        eh = max(1, int(openness * max_eh))
        for dx in range(ew):
            for dy in range(eh):
                nx = (dx - ew/2) / (ew/2)
                ny = (dy - eh/2) / max(0.5, eh/2)
                if nx*nx + ny*ny <= 1.0:
                    col = CYAN if (abs(nx)<0.3 and abs(ny)<0.3) else gc(GREEN_MID)
                    px(surf, ex+dx, ey+dy, col)
        boff = int((1.0 - genes[4]) * 3)
        for dx in range(ew-2):
            px(surf, ex+1+dx, ey-3-boff, gc(GREEN_BRIGHT))

    ol = genes[2] if tick % 150 >= 4 else 0.0
    or_ = genes[3] if tick % 150 >= 4 else 0.0
    draw_eye(cx-13, cy-8, ol)
    draw_eye(cx+5,  cy-8, or_)

    # Nariz
    px(surf, cx,   cy+2, gc(GREEN_DIM))
    px(surf, cx-1, cy+2, gc(GREEN_DIM))

    # Boca
    smile  = genes[1]
    m_open = genes[0]
    mw = 16
    mx = cx - mw//2
    my = cy + 10

    if m_open > 0.25:
        th = max(1, int(m_open * 5))
        for dx in range(1, mw-1):
            for dy in range(th):
                px(surf, mx+dx, my+dy+1,
                   (int(160*br), int(255*br), int(160*br)))

    for dx in range(mw):
        t = dx / mw
        curve = int((smile*2 - 1) * 3 * math.sin(math.pi * t))
        px(surf, mx+dx, my+curve, gc(GREEN_BRIGHT))
        if dx == 0 or dx == mw-1:
            px(surf, mx+dx, my+curve-1, gc(GREEN_MID))

    # Scanlines CRT
    scan = pygame.Surface((ANIM_W, ANIM_H), pygame.SRCALPHA)
    for y in range(0, ANIM_H, 2):
        pygame.draw.line(scan, (0, 0, 0, 55), (0, y), (ANIM_W, y))
    surf.blit(scan, (0, 0))

    # Marco
    for t in range(3):
        pygame.draw.rect(surf, (0, int(150*br), int(40*br)),
                         (t, t, ANIM_W-2*t, ANIM_H-2*t), 1)

    surf.blit(font_small.render(f"GA tick:{tick:05d}", True, GREEN_DIM), (8, 6))
    surf.blit(font_small.render("KAREN OS v2.1  |  GENETIC FACE ENGINE",
                                True, GREEN_DIM), (8, ANIM_H-18))

# ──────────────────────────────────────────────
# DESCARGA MODELO MEDIAPIPE
# ──────────────────────────────────────────────
def try_get_mediapipe_detector():
    import os, urllib.request
    model_path = "face_landmarker.task"
    model_url  = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

    if not os.path.exists(model_path):
        print(f"📥 Descargando modelo MediaPipe…")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ Modelo descargado.")
        except Exception as e:
            print(f"⚠️  No se pudo descargar: {e}")
            return None

    try:
        det = FaceDetectorMediaPipe(model_path)
        print("✅ MediaPipe FaceLandmarker listo.")
        return det
    except Exception as e:
        print(f"⚠️  MediaPipe falló: {e}")
        return None

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("🤖 KAREN ROBOT iniciando…")

    detector = None
    if NEW_MP_API:
        detector = try_get_mediapipe_detector()
    if detector is None:
        print("⚠️  Usando Haar Cascade (modo básico).")
        detector = FaceDetectorHaar()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    pygame.init()
    screen     = pygame.display.set_mode((ANIM_W, ANIM_H))
    pygame.display.set_caption("KAREN ROBOT — Genetic Face Engine")
    clock      = pygame.time.Clock()
    font_small = pygame.font.SysFont("Courier New", 11, bold=True)

    population   = [rand_ind() for _ in range(POP_SIZE)]
    smooth_genes = [0.3, 0.7, 0.8, 0.8, 0.5, 0.9]
    target       = smooth_genes[:]
    tick = 0

    print("✅ Corriendo — presiona Q para salir.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release(); pygame.quit(); sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                cap.release(); pygame.quit(); sys.exit(0)

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        target, frame = detector.detect(frame)

        population = evolve(population, target)
        best       = max(population, key=lambda i: fitness(i, target))

        alpha = 0.12
        for i in range(6):
            smooth_genes[i] = smooth_genes[i] * (1 - alpha) + best[i] * alpha

        draw_karen(screen, smooth_genes, tick, font_small)
        pygame.display.flip()

        cv2.putText(frame, "KAREN ENGINE | Q = salir",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,80), 2)
        cv2.imshow("Webcam — Deteccion Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        tick += 1
        clock.tick(FPS)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()