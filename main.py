import cv2
import numpy as np
import pygame
import pymunk
from pymunk.pygame_util import DrawOptions

# ------------------- Конфигурация -------------------
WIDTH, HEIGHT = 800, 600
RADIUS = 12
GRAVITY = 900.0          # пикселей/с² (подобрано для pymunk)
START_POS = (WIDTH // 2, 50)

# ------------------- Геометрические утилиты -------------------
def closest_point_on_segment(p, a, b):
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 == 0:
        return a, False
    t = np.dot(p - a, ab) / ab_len2
    inside = 0.0 <= t <= 1.0
    t_clamped = max(0.0, min(1.0, t))
    return a + t_clamped * ab, inside

# ------------------- Обнаружение линий (из hackathon.py) -------------------
def find_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = [pt[0] for pt in approx]
        for i in range(len(points) - 1):
            a = np.array(points[i], dtype=float)
            b = np.array(points[i+1], dtype=float)
            if np.linalg.norm(b - a) > 20:
                segments.append((a, b))
    return segments

# ------------------- Основная программа -------------------
def main():
    # --- Часть 1: захват с камеры и получение линий ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", WIDTH, HEIGHT)

    snapshot_taken = False
    lines = []

    print("Нажмите ПРОБЕЛ, чтобы сделать снимок и запустить физику")

    while not snapshot_taken:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра")
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            snapshot_taken = True
            lines = find_lines(frame)
            print(f"Снимок сделан, найдено линий: {len(lines)}")
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    # --- Часть 2: настройка Pygame и Pymunk ---
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ball on Detected Surfaces")
    clock = pygame.time.Clock()

    # Физическое пространство
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)

    # Добавляем статические отрезки (поверхности)
    for a, b in lines:
        # pymunk.Segment принимает координаты (x1,y1) и (x2,y2) в пикселях
        segment = pymunk.Segment(space.static_body, a, b, 5)   # толщина 5 пикселей
        segment.elasticity = 0.6
        segment.friction = 0.4
        space.add(segment)

    # Создаём шарик
    mass = 1.0
    radius = RADIUS
    inertia = pymunk.moment_for_circle(mass, 0, radius)
    ball_body = pymunk.Body(mass, inertia)
    ball_body.position = START_POS
    ball_shape = pymunk.Circle(ball_body, radius)
    ball_shape.elasticity = 0.7
    ball_shape.friction = 0.3
    space.add(ball_body, ball_shape)

    # Для отрисовки через Pymunk (отладочный)
    draw_options = DrawOptions(screen)

    # --- Часть 3: главный цикл физики и отрисовки ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Сброс позиции шарика
                    ball_body.position = START_POS
                    ball_body.velocity = (0, 0)
                elif event.key == pygame.K_SPACE:
                    # Принудительный сброс скорости
                    ball_body.velocity = (0, 0)

        # Шаг физики (60 fps)
        space.step(1 / 60.0)

        # Отрисовка
        screen.fill((0, 0, 0))                # чёрный фон
        # Рисуем линии вручную (красным), чтобы они отличались от отладочного режима
        for a, b in lines:
            pygame.draw.line(screen, (255, 0, 0), (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 3)
        # Рисуем шарик (жёлтый)
        pygame.draw.circle(screen, (255, 255, 0),
                           (int(ball_body.position.x), int(ball_body.position.y)),
                           radius)
        # Дополнительно можно включить отладочную отрисовку pymunk (но она перерисует шарик и линии)
        # space.debug_draw(draw_options)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()