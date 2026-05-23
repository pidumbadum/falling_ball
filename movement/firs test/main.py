"""
main.py
Пример использования BallSimulator
"""

from ball_simulator import BallSimulator, SimConfig


def setup_my_scene(sim: BallSimulator):
    """Настройка сцены: поверхности + шары"""
    # Пол
    sim.add_surface((50, 550), (750, 550))
    
    # Наклонная плоскость
    sim.add_surface((200, 450), (400, 350))
    
    # Платформа
    sim.add_surface((500, 200), (700, 200))
    
    # Стена-отбойник
    sim.add_surface((100, 100), (100, 400))
    
    # Основной шар
    sim.add_ball((100, 100), radius=20, color=(220, 50, 50))


def main():
    # Кастомная конфигурация (опционально)
    config = SimConfig(
        width=900,
        height=650,
        gravity_y=800,
        title="My Custom Physics 🎱"
    )
    
    # Создаём и запускаем симулятор
    sim = BallSimulator(config)
    sim.run(setup_callback=setup_my_scene)


if __name__ == "__main__":
    main()