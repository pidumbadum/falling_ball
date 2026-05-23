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
    """Ближайшая точка на отрезке ab к точке p"""
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = max(0.0, min(1.0, t))
    return a + t * ab

def project_vector(v, direction):
    """Проекция вектора v на направление direction (единичный вектор)"""
    dot = np.dot(v, direction)
    return direction * dot

def update_ball(lines):
    global ball_pos, ball_vel, running

    ground_line = None
    closest_pt = None
    min_dist = RADIUS

    for a, b in lines:
        closest = closest_point_on_segment(ball_pos, a, b)
        dist = np.linalg.norm(ball_pos - closest)
        if dist < min_dist:
            min_dist = dist
            ground_line = (a, b)
            closest_pt = closest

    if ground_line and closest_pt is not None:
        a, b = ground_line
        line_dir = (b - a) / np.linalg.norm(b - a)

        # Проекция скорости и ускорения
        vel_tangent = project_vector(ball_vel, line_dir)
        gravity_vec = np.array([0.0, GRAVITY])
        acc_tangent = project_vector(gravity_vec, line_dir)

        vel_tangent += acc_tangent
        vel_tangent *= 0.995  # трение

        ball_vel = vel_tangent
        ball_pos += ball_vel

        # Коррекция позиции – точно на отрезок
        ball_pos = closest_point_on_segment(ball_pos, a, b)
        # (никакого смещения по нормали!)
    else:
        ball_vel[1] += GRAVITY
        ball_pos += ball_vel

    if (ball_pos[0] < 0 or ball_pos[0] > WIDTH or
        ball_pos[1] < 0 or ball_pos[1] > HEIGHT):
        reset_ball()

def reset_ball():
    global ball_pos, ball_vel, running
    ball_pos = np.array(START_POS, dtype=float)
    ball_vel = np.array([0.0, 0.0])
    running = False

def find_lines(frame):
    """Обнаружение чёрных линий (площадок) на кадре и возврат списка отрезков"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Пороговая обработка: выделяем тёмные линии (значение порога подберите под свой фон)
    _, thresh = cv2.threshold(gray, 100, 155, cv2.THRESH_BINARY_INV)
    # Морфологическое улучшение
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for cnt in contours:
        # Аппроксимация контура полигоном
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Преобразуем в список точек (x,y)
        points = [pt[0] for pt in approx]
        # Соединяем последовательные точки в отрезки
        for i in range(len(points) - 1):
            a = np.array(points[i], dtype=float)
            b = np.array(points[i+1], dtype=float)
            if np.linalg.norm(b - a) > 20:  # отбрасываем короткие
                segments.append((a, b))
    return segments

def main():
    global running, ball_pos, ball_vel

    for i in range(5):
        temp = cv2.VideoCapture(i)
        if temp.isOpened():
            print(f"Индекс {i}: камера работает")
            temp.release()
        else:
            print(f"Индекс {i}: камера не отвечает")

    cap = cv2.VideoCapture(0)
    # Дополнительные настройки (опционально)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cv2.namedWindow("Ball Fall", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Fall", WIDTH, HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        lines = find_lines(frame)

        if running:
            update_ball(lines)

        for (a, b) in lines:
            cv2.line(frame, tuple(a.astype(int)), tuple(b.astype(int)), (0, 0, 255), 3)

        cv2.circle(frame, tuple(ball_pos.astype(int)), RADIUS, (0, 255, 255), -1)

        cv2.imshow("Ball Fall", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if not running:
                reset_ball()
                running = True
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()