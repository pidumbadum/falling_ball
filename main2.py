"""
ball_projector.py
=================
Единый файл: захват препятствий с камеры → физика pymunk → отрисовка pygame.

Два окна:
  • OpenCV  "Оператор"  — живая камера / снимок с линиями (на экране оператора)
  • pygame  "Проектор"  — только шарик на чёрном фоне    (на проекторе)

Управление (фокус на любом окне):
  ПРОБЕЛ  — сделать снимок с камеры, найти линии и запустить шарик
  R       — сбросить шарик (без нового снимка)
  ESC / Q — выход
"""

import sys
import cv2
import numpy as np
import pygame
import pymunk
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════
# 1. КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Камера
    cam_width: int = 800
    cam_height: int = 600
    camera_index: int = 0

    # Окно проектора (pygame).
    # projector_fullscreen = True  →  занимает весь второй монитор
    # projector_fullscreen = False →  обычное окно projector_width × projector_height
    projector_fullscreen: bool = False
    projector_width: int = 800
    projector_height: int = 600

    # Шарик
    ball_radius: float = 20
    ball_mass: float = 1.0
    ball_elasticity: float = 0.7
    ball_friction: float = 0.3
    start_pos: Tuple[float, float] = (400, 50)

    # Физика
    gravity_x: float = 0.0
    gravity_y: float = 900.0
    steps_per_second: int = 60

    # Поверхности
    surface_elasticity: float = 0.8
    surface_friction: float = 0.5
    surface_thickness: float = 5.0

    # Детектор линий
    min_segment_length: int = 30

    # Цвета окна оператора (OpenCV, BGR)
    op_line_color: Tuple[int, int, int] = (0, 255, 0)
    op_ball_color: Tuple[int, int, int] = (0, 255, 255)

    # Цвета окна проектора (pygame, RGB)
    proj_bg_color: Tuple[int, int, int] = (0, 0, 0)
    proj_ball_color: Tuple[int, int, int] = (255, 80, 80)


# ══════════════════════════════════════════════════════════════════
# 2. ДЕТЕКТОР ЛИНИЙ
# ══════════════════════════════════════════════════════════════════

def find_lines(frame: np.ndarray, min_length: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Обнаруживает тёмные линии на кадре, возвращает список отрезков [(a, b), ...]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segments: List[Tuple[np.ndarray, np.ndarray]] = []

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = [pt[0] for pt in approx]
        for i in range(len(points) - 1):
            a = np.array(points[i], dtype=float)
            b = np.array(points[i + 1], dtype=float)
            if np.linalg.norm(b - a) >= min_length:
                segments.append((a, b))

    return segments


# ══════════════════════════════════════════════════════════════════
# 3. ФИЗИЧЕСКИЙ СИМУЛЯТОР
# ══════════════════════════════════════════════════════════════════

class BallSimulator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.space = pymunk.Space()
        self.space.gravity = (cfg.gravity_x, cfg.gravity_y)
        self.surfaces: List[pymunk.Shape] = []
        self.balls: List[dict] = []
        self.paused = False

    def clear_surfaces(self):
        for s in self.surfaces:
            self.space.remove(s)
        self.surfaces.clear()

    def add_segment(self, p1, p2) -> pymunk.Shape:
        shape = pymunk.Segment(self.space.static_body, p1, p2, self.cfg.surface_thickness)
        shape.elasticity = self.cfg.surface_elasticity
        shape.friction = self.cfg.surface_friction
        self.space.add(shape)
        self.surfaces.append(shape)
        return shape

    def load_segments(self, segments: List[Tuple[np.ndarray, np.ndarray]]):
        self.clear_surfaces()
        for a, b in segments:
            self.add_segment(tuple(a), tuple(b))

    def clear_balls(self):
        for ball in self.balls:
            self.space.remove(ball["body"], ball["shape"])
        self.balls.clear()

    def add_ball(self, position, velocity=None) -> dict:
        cfg = self.cfg
        inertia = pymunk.moment_for_circle(cfg.ball_mass, 0, cfg.ball_radius)
        body = pymunk.Body(cfg.ball_mass, inertia)
        body.position = position
        if velocity:
            body.velocity = velocity
        shape = pymunk.Circle(body, cfg.ball_radius)
        shape.elasticity = cfg.ball_elasticity
        shape.friction = cfg.ball_friction
        self.space.add(body, shape)
        ball_data = {"body": body, "shape": shape, "radius": cfg.ball_radius}
        self.balls.append(ball_data)
        return ball_data

    def reset_ball(self):
        self.clear_balls()
        self.add_ball(self.cfg.start_pos)

    def step(self):
        if not self.paused:
            self.space.step(1.0 / self.cfg.steps_per_second)

    def get_ball_positions(self):
        return [b["body"].position for b in self.balls]

    def is_offscreen(self, margin: int = 80) -> bool:
        w, h = self.cfg.cam_width, self.cfg.cam_height
        for b in self.balls:
            x, y = b["body"].position
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                return True
        return False


# ══════════════════════════════════════════════════════════════════
# 4. ГЛАВНЫЙ ЦИКЛ
# ══════════════════════════════════════════════════════════════════

def main():
    cfg = Config()

    # ── Камера ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)

    # ── Окно оператора (OpenCV) ───────────────────────────────────
    cv2.namedWindow("Оператор", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Оператор", cfg.cam_width, cfg.cam_height)

    # ── Окно проектора (pygame) ───────────────────────────────────
    pygame.init()
    if cfg.projector_fullscreen:
        proj_screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        proj_screen = pygame.display.set_mode((cfg.projector_width, cfg.projector_height))
    pygame.display.set_caption("Проектор")
    proj_w, proj_h = proj_screen.get_size()
    clock = pygame.time.Clock()

    # ── Симулятор ─────────────────────────────────────────────────
    sim = BallSimulator(cfg)
    sim.reset_ball()

    # ── Состояние ─────────────────────────────────────────────────
    snapshot: Optional[np.ndarray] = None
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    snapshot_taken = False
    running_sim = False

    # Масштаб: физика считается в координатах камеры (cam_width × cam_height),
    # проектор может иметь другое разрешение — масштабируем при отрисовке.
    def proj_scale():
        return proj_w / cfg.cam_width, proj_h / cfg.cam_height

    while True:
        # ── Кадр с камеры ─────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра.")
            break
        frame = cv2.resize(frame, (cfg.cam_width, cfg.cam_height))

        # ── Клавиши OpenCV ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):          # ESC или Q
            break
        elif key == ord(' '):
            snapshot = frame.copy()
            segments = find_lines(snapshot, cfg.min_segment_length)
            sim.load_segments(segments)
            sim.reset_ball()
            snapshot_taken = True
            running_sim = True
            print(f"Снимок сделан. Найдено отрезков: {len(segments)}")
        elif key == ord('r'):
            sim.reset_ball()
            running_sim = snapshot_taken

        # ── Клавиши pygame ────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_sim = False
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    snapshot = frame.copy()
                    segments = find_lines(snapshot, cfg.min_segment_length)
                    sim.load_segments(segments)
                    sim.reset_ball()
                    snapshot_taken = True
                    running_sim = True
                    print(f"Снимок сделан. Найдено отрезков: {len(segments)}")
                elif event.key == pygame.K_r:
                    sim.reset_ball()
                    running_sim = snapshot_taken

        # ── Шаг физики ────────────────────────────────────────────
        if running_sim:
            sim.step()
            if sim.is_offscreen():
                sim.reset_ball()

        # ═══════════════════════════════════════════════════════════
        # ОКНО ОПЕРАТОРА — камера + линии + шарик (OpenCV)
        # ═══════════════════════════════════════════════════════════
        if snapshot_taken and snapshot is not None:
            op_frame = snapshot.copy()
            for a, b in segments:
                cv2.line(op_frame, tuple(a.astype(int)), tuple(b.astype(int)),
                         cfg.op_line_color, 2)
        else:
            op_frame = frame.copy()
            cv2.putText(op_frame, "ПРОБЕЛ — снимок и запуск",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Шарик на кадре оператора
        for pos in sim.get_ball_positions():
            cx, cy = int(pos.x), int(pos.y)
            cv2.circle(op_frame, (cx, cy), int(cfg.ball_radius), cfg.op_ball_color, -1)

        if snapshot_taken:
            cv2.putText(op_frame, f"R-сброс  ESC-выход  линий:{len(segments)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

        cv2.imshow("Оператор", op_frame)

        # ═══════════════════════════════════════════════════════════
        # ОКНО ПРОЕКТОРА — только шарик на чёрном (pygame)
        # ═══════════════════════════════════════════════════════════
        proj_screen.fill(cfg.proj_bg_color)

        sx, sy = proj_scale()
        ball_r_px = int(cfg.ball_radius * min(sx, sy))

        for pos in sim.get_ball_positions():
            draw_x = int(pos.x * sx)
            draw_y = int(pos.y * sy)
            pygame.draw.circle(proj_screen, cfg.proj_ball_color, (draw_x, draw_y), ball_r_px)
            # Блик — единственный визуальный элемент помимо шарика
            highlight_r = max(2, ball_r_px // 4)
            pygame.draw.circle(proj_screen, (255, 200, 200),
                                (draw_x - ball_r_px // 4, draw_y - ball_r_px // 4),
                                highlight_r)

        pygame.display.flip()
        clock.tick(cfg.steps_per_second)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()