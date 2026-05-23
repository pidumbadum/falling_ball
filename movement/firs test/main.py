"""
main.py
Пример использования BallSimulator
"""

from ball_simulator import BallSimulator, SimConfig


def setup_my_scene(sim: BallSimulator):
    """Настройка сцены: поверхности + шары"""
    # Пол
    sim.add_surface((50, 550), (750, 550), friction=0.8)
    
    # Наклонная плоскость
    sim.add_surface((200, 450), (400, 350), elasticity=0.9)
    
    # Платформа
    sim.add_surface((500, 200), (700, 200), friction=0.2)
    
    # Стена-отбойник
    sim.add_surface((100, 100), (100, 400), elasticity=1.0, friction=0.0)
    
    # Основной шар
    sim.add_ball((100, 100), radius=25, color=(220, 50, 50))
    
    # Дополнительный шар с начальной скоростью
    sim.add_ball((600, 150), radius=15, color=(50, 150, 220), velocity=(-100, 50))


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