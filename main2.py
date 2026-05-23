"""
ball_projector.py
=================
Единый файл: захват препятствий с камеры → физика pymunk → отрисовка pygame.

Управление:
  ПРОБЕЛ  — сделать снимок с камеры, найти линии и запустить шарик
  R       — сбросить шарик (без нового снимка)
  ESC     — выход
"""

import sys
import math
import cv2
import numpy as np
import pygame
import pymunk
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ══════════════════════════════════════════════════════════════════
# 1. КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Камера / окно
    cam_width: int = 800
    cam_height: int = 600
    camera_index: int = 0
    fullscreen: bool = False          # True — полноэкранный режим для проектора

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
    min_segment_length: int = 30      # мин. длина отрезка (px)

    # Визуал
    bg_color: Tuple[int, int, int] = (0, 0, 0)
    ball_color: Tuple[int, int, int] = (255, 80, 80)
    line_color: Tuple[int, int, int] = (0, 200, 255)
    preview_ball_color: Tuple[int, int, int] = (255, 255, 0)


# ══════════════════════════════════════════════════════════════════
# 2. ДЕТЕКТОР ЛИНИЙ (из hackathon.py)
# ══════════════════════════════════════════════════════════════════

def find_lines(frame: np.ndarray, min_length: int = 30) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Обнаруживает чёрные линии на кадре и возвращает список отрезков [(a, b), ...].
    a, b — numpy-массивы float64 с координатами концов.
    """
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
# 3. ФИЗИЧЕСКИЙ СИМУЛЯТОР (из class_ball.py)
# ══════════════════════════════════════════════════════════════════

class BallSimulator:
    """Управляет физическим пространством pymunk."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.space = pymunk.Space()
        self.space.gravity = (cfg.gravity_x, cfg.gravity_y)
        self.surfaces: List[pymunk.Shape] = []
        self.balls: List[dict] = []
        self.paused = False

    # ── Поверхности ──────────────────────────────────────────────

    def clear_surfaces(self):
        for s in self.surfaces:
            self.space.remove(s)
        self.surfaces.clear()

    def add_segment(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> pymunk.Shape:
        """Добавить статический отрезок-поверхность."""
        shape = pymunk.Segment(
            self.space.static_body,
            p1, p2,
            self.cfg.surface_thickness,
        )
        shape.elasticity = self.cfg.surface_elasticity
        shape.friction = self.cfg.surface_friction
        self.space.add(shape)
        self.surfaces.append(shape)
        return shape

    def load_segments(self, segments: List[Tuple[np.ndarray, np.ndarray]]):
        """Загрузить список отрезков (из детектора линий) как поверхности."""
        self.clear_surfaces()
        for a, b in segments:
            self.add_segment(tuple(a), tuple(b))

    # ── Шарики ───────────────────────────────────────────────────

    def clear_balls(self):
        for ball in self.balls:
            self.space.remove(ball["body"], ball["shape"])
        self.balls.clear()

    def add_ball(self, position: Tuple[float, float],
                 velocity: Optional[Tuple[float, float]] = None) -> dict:
        """Добавить динамический шар."""
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
        """Удалить все шары и добавить один на стартовую позицию."""
        self.clear_balls()
        self.add_ball(self.cfg.start_pos)

    # ── Шаг симуляции ─────────────────────────────────────────────

    def step(self):
        if not self.paused:
            dt = 1.0 / self.cfg.steps_per_second
            self.space.step(dt)

    def get_ball_positions(self) -> List[Tuple[float, float]]:
        return [b["body"].position for b in self.balls]

    def get_ball_radius(self) -> float:
        return self.cfg.ball_radius

    def is_offscreen(self, margin: int = 80) -> bool:
        w, h = self.cfg.cam_width, self.cfg.cam_height
        for b in self.balls:
            x, y = b["body"].position
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                return True
        return False


# ══════════════════════════════════════════════════════════════════
# 4. ГЛАВНЫЙ ЦИКЛ (объединение всего)
# ══════════════════════════════════════════════════════════════════

def numpy_bgr_to_pygame(frame: np.ndarray) -> pygame.Surface:
    """Конвертировать кадр OpenCV (BGR) в pygame Surface (RGB)."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # pygame ожидает (width, height, 3), numpy даёт (height, width, 3)
    frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_rgb)


def main():
    cfg = Config()

    # ── Камера ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)

    # ── pygame ────────────────────────────────────────────────────
    pygame.init()
    if cfg.fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode((cfg.cam_width, cfg.cam_height))
    pygame.display.set_caption("Ball Projector  |  ПРОБЕЛ — снимок  R — сброс  ESC — выход")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    # ── Симулятор ─────────────────────────────────────────────────
    sim = BallSimulator(cfg)
    sim.reset_ball()

    # ── Состояние ─────────────────────────────────────────────────
    snapshot: Optional[np.ndarray] = None      # снимок с камеры
    snapshot_surf: Optional[pygame.Surface] = None
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    snapshot_taken = False
    running_sim = False

    screen_w, screen_h = screen.get_size()

    while True:
        # ── Читаем кадр ───────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра.")
            break
        frame = cv2.resize(frame, (cfg.cam_width, cfg.cam_height))

        # ── События ───────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cap.release()
                    pygame.quit()
                    sys.exit()

                elif event.key == pygame.K_SPACE:
                    # Сделать снимок, найти линии, запустить симуляцию
                    snapshot = frame.copy()
                    snapshot_surf = numpy_bgr_to_pygame(snapshot)
                    segments = find_lines(snapshot, cfg.min_segment_length)
                    sim.load_segments(segments)
                    sim.reset_ball()
                    snapshot_taken = True
                    running_sim = True
                    print(f"Снимок сделан. Найдено отрезков: {len(segments)}")

                elif event.key == pygame.K_r:
                    # Только сброс шарика (линии остаются)
                    sim.reset_ball()
                    running_sim = snapshot_taken

        # ── Шаг физики ────────────────────────────────────────────
        if running_sim:
            sim.step()
            # Авто-сброс при выходе за экран
            if sim.is_offscreen():
                sim.reset_ball()

        # ── Отрисовка ─────────────────────────────────────────────
        screen.fill(cfg.bg_color)

        if snapshot_taken and snapshot_surf is not None:
            # Масштабируем снимок под размер окна pygame (нужно для полноэкранного режима)
            scaled = pygame.transform.scale(snapshot_surf, (screen_w, screen_h))
            screen.blit(scaled, (0, 0))

            # Линии поверхностей
            sx = screen_w / cfg.cam_width
            sy = screen_h / cfg.cam_height
            for a, b in segments:
                pa = (int(a[0] * sx), int(a[1] * sy))
                pb = (int(b[0] * sx), int(b[1] * sy))
                pygame.draw.line(screen, cfg.line_color, pa, pb, 3)
        else:
            # Режим предпросмотра — живая картинка с камеры
            live_surf = numpy_bgr_to_pygame(frame)
            scaled = pygame.transform.scale(live_surf, (screen_w, screen_h))
            screen.blit(scaled, (0, 0))

        # Шарики
        sx = screen_w / cfg.cam_width
        sy = screen_h / cfg.cam_height
        ball_r_px = int(cfg.ball_radius * min(sx, sy))
        for pos in sim.get_ball_positions():
            draw_x = int(pos.x * sx)
            draw_y = int(pos.y * sy)
            # Тень
            pygame.draw.circle(screen, (30, 30, 30), (draw_x + 3, draw_y + 4), ball_r_px)
            # Шарик
            pygame.draw.circle(screen, cfg.ball_color, (draw_x, draw_y), ball_r_px)
            # Блик
            pygame.draw.circle(screen, (255, 200, 200),
                                (draw_x - ball_r_px // 4, draw_y - ball_r_px // 4),
                                max(2, ball_r_px // 4))

        # Подсказка
        if not snapshot_taken:
            hint = font.render("ПРОБЕЛ — снимок с камеры и запуск шарика", True, (220, 220, 220))
            screen.blit(hint, (10, 10))
        else:
            hint = font.render(f"R — сброс   ESC — выход   линий: {len(segments)}", True, (180, 180, 180))
            screen.blit(hint, (10, 10))

        pygame.display.flip()
        clock.tick(cfg.steps_per_second)


if __name__ == "__main__":
    main()