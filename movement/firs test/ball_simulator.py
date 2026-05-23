"""
ball_simulator.py
Необходим только для того, чтобы обрабатывать события шара 
"""
import pymunk
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SimConfig:
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
    
    offscreen_buffer: int = 50


class BallSimulator:
    """Контроллер физической симуляции"""
    
    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        
        # Физическое пространство
        self.space = pymunk.Space()
        self.space.gravity = (self.config.gravity_x, self.config.gravity_y)
        
        # Хранилища объектов
        self.surfaces: List[pymunk.Shape] = []
        self.balls: List[dict] = []  # [{'body': Body, 'shape': Circle, 'radius': ...}]
        
        # Флаги управления симуляцией
        self.paused = False

    def add_surface(self, p1: Tuple[float, float], p2: Tuple[float, float],
                    elasticity: Optional[float] = None,
                    friction: Optional[float] = None,
                    thickness: Optional[float] = None) -> pymunk.Shape:
        """Добавить статическую поверхность (отрезок)"""
        elastic = elasticity if elasticity is not None else self.config.default_surface_elasticity
        fric = friction if friction is not None else self.config.default_surface_friction
        thick = thickness if thickness is not None else self.config.default_surface_thickness
        
        shape = pymunk.Segment(self.space.static_body, p1, p2, thick)
        shape.elasticity = elastic
        shape.friction = fric
        # shape.color удалён (нужен только для debug_draw)
        
        self.space.add(shape)
        self.surfaces.append(shape)
        return shape
    
    def add_ball(self, position: Tuple[float, float],
                 radius: Optional[float] = None,
                 mass: Optional[float] = None,
                 elasticity: Optional[float] = None,
                 friction: Optional[float] = None,
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
        # shape.color удалён
        
        self.space.add(body, shape)
        ball_data = {'body': body, 'shape': shape, 'radius': rad}
        self.balls.append(ball_data)
        return ball_data
    
    def add_polygon_surface(self, points: List[Tuple[float, float]],
                           elasticity: Optional[float] = None,
                           friction: Optional[float] = None) -> pymunk.Shape:
        """Добавить поверхность в виде многоугольника"""
        elastic = elasticity if elasticity is not None else self.config.default_surface_elasticity
        fric = friction if friction is not None else self.config.default_surface_friction
        
        shape = pymunk.Poly(self.space.static_body, points)
        shape.elasticity = elastic
        shape.friction = fric
        # shape.color удалён
        
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
        """Получить позицию центра шара"""
        if 0 <= ball_index < len(self.balls):
            return self.balls[ball_index]['body'].position
        return None
    
    def step(self):
        """Выполнить один шаг физики (dt = 1 / steps_per_second)"""
        if not self.paused:
            self.space.step(1 / self.config.steps_per_second)
    
    
    def reset_simulation(self):
        """Сброс симуляции (удаляет все шары, оставляет поверхности)"""
        for ball in self.balls:
            self.space.remove(ball['body'], ball['shape'])
        self.balls.clear()