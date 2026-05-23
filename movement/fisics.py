import pygame
import pymunk
from pymunk.pygame_util import DrawOptions

# --- Настройка Pygame ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ball on Surfaces")
clock = pygame.time.Clock()

# --- Настройка Pymunk (физическое пространство) ---
space = pymunk.Space()
space.gravity = (0, 900)  # Гравитация вниз (пиксели/с²)
draw_options = DrawOptions(screen)

# --- Функция создания поверхности (отрезка) ---
def add_surface(p1, p2, elasticity=0.8, friction=0.5):
    shape = pymunk.Segment(space.static_body, p1, p2, 5)  # 5 = толщина/радиус столкновения
    shape.elasticity = elasticity  # Упругость (0..1)
    shape.friction = friction      # Трение (0..1)
    space.add(shape)

# --- Создаём поверхности ---
# Пол, наклонная плоскость, платформа
add_surface((50, 550), (750, 550))
add_surface((200, 450), (400, 350))
add_surface((500, 200), (700, 200))

# --- Создаём шар ---
mass, radius = 1, 20
inertia = pymunk.moment_for_circle(mass, 0, radius)
ball_body = pymunk.Body(mass, inertia)
ball_body.position = (100, 100)
ball_shape = pymunk.Circle(ball_body, radius)
ball_shape.elasticity = 0.7
ball_shape.friction = 0.3
space.add(ball_body, ball_shape)

# --- Основной цикл ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Клик перемещает шар в точку клика и сбрасывает скорость
            ball_body.position = event.pos
            ball_body.velocity = (0, 0)

    # Шаг физики (фиксированный шаг для стабильности)
    space.step(1 / 60)

    # Отрисовка
    screen.fill((240, 240, 250))
    space.debug_draw(draw_options)  # Рисует все тела и формы
    pygame.display.flip()
    clock.tick(60)

pygame.quit()