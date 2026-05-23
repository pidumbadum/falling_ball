import pygame
from ball_simulator import BallSimulator, SimConfig

def main():
    # 1. Инициализация Pygame
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Physics Renderer")
    clock = pygame.time.Clock()
    FPS = 60

    # 2. Создание физического симулятора
    cfg = SimConfig(gravity_y=900, steps_per_second=60)
    sim = BallSimulator(cfg)

    # 3. Настройка сцены (вызывается один раз)
    sim.add_surface((0, 550), (WIDTH, 550), thickness=8)
    sim.add_surface((200, 400), (600, 300), thickness=5)
    sim.add_ball((400, 100), velocity=(150, 0))

    # Цвета для отрисовки (так как в симуляторе их больше нет)
    BALL_COLOR = (220, 60, 60)
    SURFACE_COLOR = (70, 70, 80)
    BG_COLOR = (240, 240, 250)

    # 4. Главный цикл
    running = True
    while running:
        # Обработка ввода
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_r:
                    sim.reset_simulation()
                    # Пересоздаём сцену после сброса
                    sim.add_surface((0, 550), (WIDTH, 550), thickness=8)
                    sim.add_surface((200, 400), (600, 300), thickness=5)
                    sim.add_ball((400, 100), velocity=(150, 0))

        # Обновление физики
        sim.step()

        # Отрисовка
        screen.fill(BG_COLOR)

        # Рисуем поверхности (берём точки из pymunk.Shape)
        for shape in sim.surfaces:
            if hasattr(shape, 'a'):  # pymunk.Segment
                p1 = shape.a
                p2 = shape.b
                pygame.draw.line(screen, SURFACE_COLOR, p1, p2, int(shape.radius * 2))
            elif hasattr(shape, 'get_vertices'):  # pymunk.Poly
                verts = shape.get_vertices()
                if len(verts) >= 3:
                    pygame.draw.polygon(screen, SURFACE_COLOR, verts)

        # Рисуем шары
        for ball in sim.balls:
            pos = ball['body'].position
            rad = ball['radius']
            # Pygame draw.circle принимает целые координаты
            pygame.draw.circle(screen, BALL_COLOR, (int(pos.x), int(pos.y)), int(rad))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()