import numpy as np


def get_preset_obstacles(preset_type, screen_w, screen_h, playable_side, left_margin):
    """
    Возвращает масштабируемый массив стен под текущее разрешение монитора.
    Стены упрощены по длине, чтобы пчелы успевали преодолеть путь за 500 кадров жизни.
    """
    if preset_type == 1:
        # Пресет 1: Центральный широкий блок
        return np.array([[
            0.2 * screen_w, 0.45 * screen_h,
            0.8 * screen_w, 0.45 * screen_h
        ]], dtype=np.float32)

    elif preset_type == 2:
        # Пресет 2: Двойная щель во всю ширину экрана
        return np.array([
            [0, 0.5 * screen_h, 0.38 * screen_w, 0.5 * screen_h],
            [0.46 * screen_w, 0.5 * screen_h, 0.54 * screen_w, 0.5 * screen_h],
            [0.62 * screen_w, 0.5 * screen_h, screen_w, 0.5 * screen_h]
        ], dtype=np.float32)

    elif preset_type == 3:
        # Пресет 3: Упрощенный зигзаг
        return np.array([
            [0.10 * screen_w, 0.35 * screen_h, 0.60 * screen_w, 0.35 * screen_h],
            [0.50 * screen_w, 0.65 * screen_h, 0.90 * screen_w, 0.65 * screen_h]
        ], dtype=np.float32)

    elif preset_type == 4:
        # Пресет 4: Воронка (диагональные стены, направляющие рой в узкое горлышко)
        return np.array([
            [0, 0.35 * screen_h, 0.44 * screen_w, 0.55 * screen_h],
            [screen_w, 0.35 * screen_h, 0.56 * screen_w, 0.55 * screen_h]
        ], dtype=np.float32)

    elif preset_type == 5:
        # Пресет 5: Упрощенный трехуровневый лабиринт
        return np.array([
            [0, 0.3 * screen_h, 0.5 * screen_w, 0.3 * screen_h],
            [0.4 * screen_w, 0.5 * screen_h, screen_w, 0.5 * screen_h],
            [0, 0.7 * screen_h, 0.5 * screen_w, 0.7 * screen_h]
        ], dtype=np.float32)

    return np.zeros((0, 4), dtype=np.float32)