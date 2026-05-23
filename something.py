# -*- coding: cp1251 -*-
"""
ball_projector.py
=================
Захват препятствий с камеры (с выделением ROI) → физика pymunk → отрисовка pygame.

Два окна:
  • OpenCV  "Оператор"  — живая камера, выделение ROI, снимок с линиями + шарик
  • pygame  "Проектор"  — только шарик на чёрном фоне (для проектора)

Управление:
  Мышь     — нарисовать прямоугольник ROI (зону с препятствиями)
  ПРОБЕЛ   — сделать снимок, найти линии в ROI и запустить шарик
  R        — сбросить шарик
  C        — сбросить ROI (выделить заново)
  ESC / Q  — выход

Координаты физики считаются в пространстве ПРОЕКТОРА (projector_width × projector_height).
ROI на кадре камеры масштабируется в это пространство — шарик не обрезается.
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
    camera_index: int = 1             # индекс камеры (1 + CAP_DSHOW на Windows)
    use_dshow: bool = True            # True — использовать DirectShow (Windows)

    # Окно проектора (pygame)
    projector_fullscreen: bool = False
    projector_width: int = 800        # координатное пространство физики и проектора
    projector_height: int = 600

    # Шарик — стартовая позиция задаётся в долях экрана проектора
    ball_start_x_frac: float = 0.5   # 0.0 = левый край, 1.0 = правый
    ball_start_y_frac: float = 0.05  # почти у верхнего края
    ball_radius: float = 20
    ball_mass: float = 1.0
    ball_elasticity: float = 0.7
    ball_friction: float = 0.3

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

    # Цвета оператора (OpenCV BGR)
    op_roi_color: Tuple[int, int, int] = (0, 255, 255)
    op_line_color: Tuple[int, int, int] = (0, 255, 0)
    op_ball_color: Tuple[int, int, int] = (0, 100, 255)

    # Цвета проектора (pygame RGB)
    proj_bg_color: Tuple[int, int, int] = (0, 0, 0)
    proj_ball_color: Tuple[int, int, int] = (255, 80, 80)

    @property
    def ball_start_pos(self) -> Tuple[float, float]:
        return (self.projector_width * self.ball_start_x_frac,
                self.projector_height * self.ball_start_y_frac)


# ══════════════════════════════════════════════════════════════════
# 2. ДЕТЕКТОР ЛИНИЙ
# ══════════════════════════════════════════════════════════════════

def find_lines_in_roi(
    frame: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],   # (x1, y1, x2, y2) в пикселях камеры
    proj_w: int,
    proj_h: int,
    min_length: int = 30,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Ищет тёмные линии внутри ROI на кадре камеры.
    Возвращает отрезки [(a, b)] в координатах ПРОЕКТОРА (proj_w × proj_h).
    Если ROI не задан — использует весь кадр.
    """
    h_cam, w_cam = frame.shape[:2]

    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, x2 = sorted([max(0, x1), min(w_cam, x2)])
        y1, y2 = sorted([max(0, y1), min(h_cam, y2)])
        crop = frame[y1:y2, x1:x2]
        roi_x, roi_y = x1, y1
        roi_w, roi_h = x2 - x1, y2 - y1
    else:
        crop = frame
        roi_x, roi_y = 0, 0
        roi_w, roi_h = w_cam, h_cam

    if crop.size == 0:
        return []

    # Детекция
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segments: List[Tuple[np.ndarray, np.ndarray]] = []

    # Масштаб: ROI-пиксели → координаты проектора
    sx = proj_w / roi_w
    sy = proj_h / roi_h
    # Смещение ROI внутри кадра → тоже в координаты проектора
    ox = roi_x * (proj_w / w_cam)
    oy = roi_y * (proj_h / h_cam)

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = [pt[0] for pt in approx]
        for i in range(len(points) - 1):
            # Точки в координатах crop → координаты проектора
            a = np.array([points[i][0] * sx + ox,
                           points[i][1] * sy + oy], dtype=float)
            b = np.array([points[i+1][0] * sx + ox,
                           points[i+1][1] * sy + oy], dtype=float)
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

    def add_segment(self, p1, p2):
        shape = pymunk.Segment(self.space.static_body, p1, p2, self.cfg.surface_thickness)
        shape.elasticity = self.cfg.surface_elasticity
        shape.friction = self.cfg.surface_friction
        self.space.add(shape)
        self.surfaces.append(shape)

    def load_segments(self, segments):
        self.clear_surfaces()
        for a, b in segments:
            self.add_segment(tuple(a), tuple(b))

    def clear_balls(self):
        for ball in self.balls:
            self.space.remove(ball["body"], ball["shape"])
        self.balls.clear()

    def add_ball(self, position, velocity=None):
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
        self.balls.append({"body": body, "shape": shape})

    def reset_ball(self):
        self.clear_balls()
        self.add_ball(self.cfg.ball_start_pos)

    def step(self):
        if not self.paused:
            self.space.step(1.0 / self.cfg.steps_per_second)

    def get_ball_positions(self):
        return [b["body"].position for b in self.balls]

    def is_offscreen(self, margin: int = 100) -> bool:
        w, h = self.cfg.projector_width, self.cfg.projector_height
        for b in self.balls:
            x, y = b["body"].position
            if x < -margin or x > w + margin or y < -margin or y > h + margin:
                return True
        return False


# ══════════════════════════════════════════════════════════════════
# 4. ВЫДЕЛЕНИЕ ROI МЫШЬЮ
# ══════════════════════════════════════════════════════════════════

class ROISelector:
    """Позволяет пользователю нарисовать прямоугольник мышью на окне OpenCV."""

    def __init__(self):
        self.drawing = False
        self.start: Optional[Tuple[int, int]] = None
        self.end: Optional[Tuple[int, int]] = None
        self.confirmed: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.end = (x, y)
            self.confirmed = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)
            x1, y1 = self.start
            x2, y2 = self.end
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.confirmed = (min(x1, x2), min(y1, y2),
                                  max(x1, x2), max(y1, y2))

    def draw_on(self, frame: np.ndarray, color=(0, 255, 255)):
        """Нарисовать текущий прямоугольник на кадре."""
        roi = self.confirmed or (
            (self.start[0], self.start[1], self.end[0], self.end[1])
            if self.drawing and self.start and self.end else None
        )
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    def reset(self):
        self.drawing = False
        self.start = self.end = self.confirmed = None


# ══════════════════════════════════════════════════════════════════
# 5. ГЛАВНЫЙ ЦИКЛ
# ══════════════════════════════════════════════════════════════════

def main():
    cfg = Config()

    # ── Камера ────────────────────────────────────────────────────
    cam_id = cfg.camera_index + cv2.CAP_DSHOW if cfg.use_dshow else cfg.camera_index
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)

    # ── Окно оператора ────────────────────────────────────────────
    WIN = "Оператор"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cfg.cam_width, cfg.cam_height)

    roi_sel = ROISelector()
    cv2.setMouseCallback(WIN, roi_sel.mouse_callback)

    # ── Окно проектора (pygame) — только шарик на чёрном ─────────
    pygame.init()
    if cfg.projector_fullscreen:
        proj_screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        proj_screen = pygame.display.set_mode(
            (cfg.projector_width, cfg.projector_height))
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

    # Масштаб: координаты проектора → пиксели окна оператора (для отрисовки шарика у оператора)
    def op_scale():
        h, w = frame.shape[:2]
        return w / cfg.projector_width, h / cfg.projector_height

    # Масштаб: координаты проектора → пиксели окна pygame
    sx_proj = proj_w / cfg.projector_width
    sy_proj = proj_h / cfg.projector_height

    frame = np.zeros((cfg.cam_height, cfg.cam_width, 3), dtype=np.uint8)

    while True:
        # ── Кадр с камеры ─────────────────────────────────────────
        ret, raw = cap.read()
        if ret:
            frame = cv2.resize(raw, (cfg.cam_width, cfg.cam_height))

        # ── Клавиши OpenCV ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            snapshot = frame.copy()
            segments = find_lines_in_roi(
                snapshot,
                roi_sel.confirmed,
                cfg.projector_width, cfg.projector_height,
                cfg.min_segment_length,
            )
            sim.load_segments(segments)
            sim.reset_ball()
            snapshot_taken = True
            running_sim = True
            print(f"Снимок. ROI={roi_sel.confirmed}. Отрезков: {len(segments)}")
        elif key == ord('r'):
            sim.reset_ball()
            running_sim = snapshot_taken
        elif key == ord('c'):
            roi_sel.reset()
            snapshot_taken = False
            running_sim = False
            sim.reset_ball()

        # ── Клавиши pygame ────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    snapshot = frame.copy()
                    segments = find_lines_in_roi(
                        snapshot,
                        roi_sel.confirmed,
                        cfg.projector_width, cfg.projector_height,
                        cfg.min_segment_length,
                    )
                    sim.load_segments(segments)
                    sim.reset_ball()
                    snapshot_taken = True
                    running_sim = True
                elif event.key == pygame.K_r:
                    sim.reset_ball()
                    running_sim = snapshot_taken

        # ── Шаг физики ────────────────────────────────────────────
        if running_sim:
            sim.step()
            if sim.is_offscreen():
                sim.reset_ball()

        # ══════════════════════════════════════════════════════════
        # ОКНО ОПЕРАТОРА (OpenCV)
        # ══════════════════════════════════════════════════════════
        if snapshot_taken and snapshot is not None:
            op_frame = snapshot.copy()
            # Рисуем найденные линии (в координатах камеры)
            h_op, w_op = op_frame.shape[:2]
            lsx = w_op / cfg.projector_width
            lsy = h_op / cfg.projector_height
            for a, b in segments:
                pa = (int(a[0] * lsx), int(a[1] * lsy))
                pb = (int(b[0] * lsx), int(b[1] * lsy))
                cv2.line(op_frame, pa, pb, cfg.op_line_color, 2)
        else:
            op_frame = frame.copy()

        # ROI прямоугольник
        roi_sel.draw_on(op_frame, cfg.op_roi_color)

        # Шарик у оператора
        h_op, w_op = op_frame.shape[:2]
        osx = w_op / cfg.projector_width
        osy = h_op / cfg.projector_height
        for pos in sim.get_ball_positions():
            cx = int(pos.x * osx)
            cy = int(pos.y * osy)
            cv2.circle(op_frame, (cx, cy), int(cfg.ball_radius * min(osx, osy)),
                       cfg.op_ball_color, -1)

        # Подсказки
        if not snapshot_taken:
            if roi_sel.confirmed:
                tip = "ROI выделен. ПРОБЕЛ — запуск  C — сбросить ROI"
            else:
                tip = "Выделите зону мышью, затем нажмите ПРОБЕЛ"
        else:
            tip = f"R-сброс  C-новый ROI  ESC-выход  линий:{len(segments)}"
        cv2.putText(op_frame, tip, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

        cv2.imshow(WIN, op_frame)

        # ══════════════════════════════════════════════════════════
        # ОКНО ПРОЕКТОРА (pygame) — только шарик на чёрном
        # ══════════════════════════════════════════════════════════
        proj_screen.fill(cfg.proj_bg_color)

        ball_r_px = int(cfg.ball_radius * min(sx_proj, sy_proj))
        for pos in sim.get_ball_positions():
            dx = int(pos.x * sx_proj)
            dy = int(pos.y * sy_proj)
            pygame.draw.circle(proj_screen, cfg.proj_ball_color, (dx, dy), ball_r_px)
            # Блик
            pygame.draw.circle(proj_screen, (255, 200, 200),
                                (dx - ball_r_px // 4, dy - ball_r_px // 4),
                                max(2, ball_r_px // 4))

        pygame.display.flip()
        clock.tick(cfg.steps_per_second)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()