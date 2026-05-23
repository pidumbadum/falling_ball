import cv2
import numpy as np

# --- Параметры ---
WIDTH, HEIGHT = 800, 600
RADIUS = 12
GRAVITY = 0.25
START_POS = (WIDTH // 2, 50)

# --- Глобальные переменные ---
ball_pos = np.array(START_POS, dtype=float)
ball_vel = np.array([0.0, 0.0])
running = False

# --- Функции геометрии ---
def closest_point_on_segment(p, a, b):
    """Ближайшая точка на отрезке ab к точке p, а также флаг, лежит ли проекция внутри отрезка"""
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 == 0:
        return a, False
    t = np.dot(p - a, ab) / ab_len2
    inside = 0.0 <= t <= 1.0
    t_clamped = max(0.0, min(1.0, t))
    return a + t_clamped * ab, inside

def project_vector(v, direction):
    """Проекция вектора v на направление direction (единичный вектор)"""
    dot = np.dot(v, direction)
    return direction * dot

def update_ball(lines):
    global ball_pos, ball_vel, running

    ground_line = None
    closest_pt = None
    min_dist = RADIUS
    on_vertex = False

    for a, b in lines:
        closest, inside = closest_point_on_segment(ball_pos, a, b)
        dist = np.linalg.norm(ball_pos - closest)
        if dist < min_dist:
            min_dist = dist
            ground_line = (a, b)
            closest_pt = closest
            on_vertex = not inside   # если проекция вне отрезка – это вершина

    if ground_line is not None and closest_pt is not None:
        a, b = ground_line
        line_dir = (b - a) / np.linalg.norm(b - a)

        if on_vertex:
            # Находим нормаль к обеим линиям, сходящимся в вершине (упрощённо – отскок)
            # В данном случае просто останавливаем тангенциальную скорость
            ball_vel = np.array([0.0, 0.0])
            ball_pos = closest_pt
        else:
            # Скатывание вдоль линии
            vel_tangent = project_vector(ball_vel, line_dir)
            gravity_vec = np.array([0.0, GRAVITY])
            acc_tangent = project_vector(gravity_vec, line_dir)

            vel_tangent += acc_tangent
            vel_tangent *= 0.995  # трение

            ball_vel = vel_tangent
            ball_pos += ball_vel
            # Прижимаем обратно к отрезку (без смещения по нормали)
            ball_pos, _ = closest_point_on_segment(ball_pos, a, b)
    else:
        # Свободное падение
        ball_vel[1] += GRAVITY
        ball_pos += ball_vel

    # Проверка выхода за пределы экрана
    if (ball_pos[0] < -RADIUS or ball_pos[0] > WIDTH + RADIUS or
        ball_pos[1] < -RADIUS or ball_pos[1] > HEIGHT + RADIUS):
        reset_ball()

def reset_ball():
    global ball_pos, ball_vel, running
    ball_pos = np.array(START_POS, dtype=float)
    ball_vel = np.array([0.0, 0.0])
    running = False

def find_lines(frame):
    """Обнаружение чёрных линий (площадок) на кадре и возврат списка отрезков"""
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

def main():
    global running, ball_pos, ball_vel

    # --- Захват камеры ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cv2.namedWindow("Ball Fall", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Fall", WIDTH, HEIGHT)

    snapshot = None
    lines_snapshot = []
    snapshot_taken = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра")
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Если ещё не сделан снимок, показываем видео с камеры
        if not snapshot_taken:
            display_frame = frame.copy()
        else:
            display_frame = snapshot.copy()

        # Рисуем линии (если снимок сделан – рисуем найденные на нём линии)
        if snapshot_taken:
            for (a, b) in lines_snapshot:
                cv2.line(display_frame, tuple(a.astype(int)), tuple(b.astype(int)), (0, 0, 255), 3)

        # Рисуем шарик
        cv2.circle(display_frame, tuple(ball_pos.astype(int)), RADIUS, (0, 255, 255), -1)

        cv2.imshow("Ball Fall", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if not running and not snapshot_taken:
                # Делаем скриншот текущего кадра
                snapshot = frame.copy()
                lines_snapshot = find_lines(snapshot)
                snapshot_taken = True
                # Сбрасываем и запускаем мячик
                reset_ball()
                running = True
                print(f"Снимок сделан, найдено линий: {len(lines_snapshot)}")
        elif key == 27:  # ESC
            break

        # Если мячик запущен и снимок есть, обновляем его положение по статичным линиям
        if running and snapshot_taken:
            update_ball(lines_snapshot)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()