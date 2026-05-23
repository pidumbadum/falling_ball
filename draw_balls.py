import pygame
import sys
import math


def draw_ball_fullscreen(positions, radius=20, speed=1000):
    """
    Открывает полноэкранное окно и плавно перемещает красный шарик по списку координат.
    :param positions: список кортежей [(x1, y1), (x2, y2), ...]
    :param radius: радиус шарика
    :param speed: скорость перемещения в пикселях в секунду
    """
    if not positions:
        return

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen_w, screen_h = screen.get_size()
    clock = pygame.time.Clock()

    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    running = True

    # Начинаем с первой точки
    curr_x, curr_y = float(positions[0][0]), float(positions[0][1])
    idx = 1  # Индекс следующей цели

    while running:
        # dt = время прошедшее с прошлого кадра (в секундах). Делает анимацию плавной на любом ПК
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # 🎯 Определяем цель
        if idx < len(positions):
            tgt_x, tgt_y = float(positions[idx][0]), float(positions[idx][1])
        else:
            # Координаты закончились → цель = текущая позиция (шарик замирает)
            tgt_x, tgt_y = curr_x, curr_y

        # 📐 Вектор и расстояние до цели
        dx = tgt_x - curr_x
        dy = tgt_y - curr_y
        dist = math.hypot(dx, dy)

        # 🚀 Движение
        if dist > 2:  # 2px - допустимая погрешность, чтобы шарик не "дрожал" в конечной точке
            move_step = speed * dt
            curr_x += (dx / dist) * move_step
            curr_y += (dy / dist) * move_step
        else:
            # Дошли до цели
            curr_x, curr_y = tgt_x, tgt_y
            if idx < len(positions):
                idx += 1  # Переключаемся на следующую точку

        # 🛡️ Ограничиваем границами экрана
        curr_x = max(radius, min(curr_x, screen_w - radius))
        curr_y = max(radius, min(curr_y, screen_h - radius))

        screen.fill(BLACK)
        pygame.draw.circle(screen, RED, (int(curr_x), int(curr_y)), radius)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


# 🔹 Пример вызова
if __name__ == "__main__":
    coords_list = [(x, y) for x in range(100, 700, 100) for y in range(100, 500, 100)]
    # speed=400 означает 400 пикселей в секунду. Меняйте под свою задачу.
    draw_ball_fullscreen(coords_list, radius=20, speed=350)