"""
ball_simulator.py
Класс для 2D-симуляции движения шаров по заданным поверхностям.
Использует pymunk (физика) + pygame (отрисовка).
"""

import pygame
import pymunk
from pymunk.pygame_util import DrawOptions
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class SimConfig:
    width: int = 800
    height: int = 600
    fps: int = 60
    title: str = "Ball Physics Simulator"
    bg_color: Tuple[int, int, int] = (240, 240, 250)
    
    gravity_x: float = 0
    gravity_y: float = 900
    steps_per_second: int = 60
    
    default_surface_elasticity: float = 0.8
    default_surface_friction: float = 0.5
    default_surface_thickness: float = 5
    
    default_ball_mass: float = 1
    default_ball_radius: float = 20
    default_ball_elasticity: float = 0.7
    default_ball_friction: float = 0.3
    
    offscreen_buffer: int = 50  # 🆕 Буфер удаления за краем экрана


class BallSimulator:
    """Основной класс симуляции"""
    
    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        
        # Инициализация Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption(self.config.title)
        self.clock = pygame.time.Clock()
        
        # Физическое пространство
        self.space = pymunk.Space()
        self.space.gravity = (self.config.gravity_x, self.config.gravity_y)
        
        # Отрисовка
        self.draw_options = DrawOptions(self.screen)
        
        # Списки объектов
        self.surfaces: List[pymunk.Shape] = []
        self.balls: List[dict] = []  # [{'body': Body, 'shape': Circle, 'color': ...}]
        
        # Флаги
        self.running = False
        self.paused = False
        
    def add_surface(self, p1: Tuple[float, float], p2: Tuple[float, float],
                    elasticity: Optional[float] = None,
                    friction: Optional[float] = None,
                    thickness: Optional[float] = None,
                    color: Tuple[int, int, int] = (50, 50, 50)) -> pymunk.Shape:
        """Добавить статическую поверхность (отрезок)"""
        elastic = elasticity if elasticity is not None else self.config.default_surface_elasticity
        fric = friction if friction is not None else self.config.default_surface_friction
        thick = thickness if thickness is not None else self.config.default_surface_thickness
        
        shape = pymunk.Segment(self.space.static_body, p1, p2, thick)
        shape.elasticity = elastic
        shape.friction = fric
        shape.color = color + (255,)  # RGBA для pymunk
        self.space.add(shape)
        self.surfaces.append(shape)
        return shape
    
    def add_ball(self, position: Tuple[float, float],
                 radius: Optional[float] = None,
                 mass: Optional[float] = None,
                 elasticity: Optional[float] = None,
                 friction: Optional[float] = None,
                 color: Tuple[int, int, int] = (200, 60, 60),
                 velocity: Optional[Tuple[float, float]] = None) -> dict:
        """Добавить динамический шар"""
        rad = radius if radius is not None else self.config.default_ball_radius
        m = mass if mass is not None else self.config.default_ball_mass
        elastic = elasticity if elasticity is not None else self.config.default_ball_elasticity
        fric = friction if friction is not None else self.config.default_ball_friction
        
        inertia = pymunk.moment_for_circle(m, 0, rad)
        body = pymunk.Body(m, inertia)
        body.position = position
        if velocity:
            body.velocity = velocity
            
        shape = pymunk.Circle(body, rad)
        shape.elasticity = elastic
        shape.friction = fric
        shape.color = color + (255,)
        
        self.space.add(body, shape)
        ball_data = {'body': body, 'shape': shape, 'color': color, 'radius': rad}
        self.balls.append(ball_data)
        return ball_data
    
    def add_polygon_surface(self, points: List[Tuple[float, float]],
                           elasticity: Optional[float] = None,
                           friction: Optional[float] = None,
                           color: Tuple[int, int, int] = (50, 50, 50)) -> pymunk.Shape:
        """Добавить поверхность в виде многоугольника"""
        elastic = elasticity if elasticity is not None else self.config.default_surface_elasticity
        fric = friction if friction is not None else self.config.default_surface_friction
        
        shape = pymunk.Poly(self.space.static_body, points)
        shape.elasticity = elastic
        shape.friction = fric
        shape.color = color + (255,)
        self.space.add(shape)
        self.surfaces.append(shape)
        return shape
    
    def apply_force_to_ball(self, ball_index: int, force: Tuple[float, float]):
        """Применить силу к шару по индексу"""
        if 0 <= ball_index < len(self.balls):
            self.balls[ball_index]['body'].apply_force_at_local_point(force, (0, 0))
    
    def set_ball_velocity(self, ball_index: int, velocity: Tuple[float, float]):
        """Задать скорость шару"""
        if 0 <= ball_index < len(self.balls):
            self.balls[ball_index]['body'].velocity = velocity
    
    def get_ball_position(self, ball_index: int) -> Optional[Tuple[float, float]]:
        """Получить позицию шара"""
        if 0 <= ball_index < len(self.balls):
            return self.balls[ball_index]['body'].position
        return None
    
    def _handle_events(self):
        """Обработка событий Pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self._reset_simulation()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Клик ЛКМ: переместить первый шар в точку клика
                if event.button == 1 and self.balls:
                    self.balls[0]['body'].position = event.pos
                    self.balls[0]['body'].velocity = (0, 0)
                # Клик ПКМ: добавить новый шар в точку клика
                elif event.button == 3:
                    self.add_ball(event.pos, color=(60, 180, 60))
    
    def _update(self):
        """Шаг физики"""
        if not self.paused:
            self.space.step(1 / self.config.steps_per_second)
    
    def _render(self):
        """Отрисовка кадра"""
        self.screen.fill(self.config.bg_color)
        self.space.debug_draw(self.draw_options)
        
        # Подсказки на экране
        self._draw_hints()
        
        pygame.display.flip()
    
    def _draw_hints(self):
        """Отрисовка подсказок"""
        font = pygame.font.Font(None, 24)
        hints = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Balls: {len(self.balls)}",
            f"Paused: {'YES' if self.paused else 'NO'}",
            "Controls: SPACE=pause, R=reset, ESC=exit",
            "LClick=move ball, RClick=add ball"
        ]
        for i, text in enumerate(hints):
            surf = font.render(text, True, (30, 30, 30))
            self.screen.blit(surf, (10, 10 + i * 20))
    
    def _reset_simulation(self):
        """Сброс симуляции (удаляет все шары, оставляет поверхности)"""
        for ball in self.balls:
            self.space.remove(ball['body'], ball['shape'])
        self.balls.clear()
    
    def run(self, setup_callback=None):
        """
        Запуск главного цикла симуляции.
        :param setup_callback: функция, которая вызывается после инициализации
                               для добавления объектов (принимает симулятор как аргумент)
        """
        if setup_callback:
            setup_callback(self)
        
        self.running = True
        while self.running:
            self._handle_events()
            self._update()
            self._render()
            self.clock.tick(self.config.fps)
        
        pygame.quit()
    
    def cleanup(self):
        """Корректное завершение"""
        pygame.quit()