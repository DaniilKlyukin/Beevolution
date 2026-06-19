import numpy as np
from collections import deque
import config


def compute_grid_distances(obstacles, goal, screen_w, screen_h, grid_size=(50, 50)):
    """
    Строит сетку расстояний от цели до всех свободных ячеек игрового поля с помощью BFS.
    Учитывает как пресеты, так и нарисованные пользователем стены.
    """
    cols, rows = grid_size
    cell_w = screen_w / cols
    cell_h = screen_h / rows

    # Заполняем сетку бесконечными расстояниями по умолчанию
    dist_grid = np.full((cols, rows), np.inf, dtype=np.float32)

    # Определяем ячейку, в которой находится цель (Дого)
    goal_col = int(goal[0] / cell_w)
    goal_row = int(goal[1] / cell_h)
    goal_col = max(0, min(cols - 1, goal_col))
    goal_row = max(0, min(rows - 1, goal_row))

    # Сетка заблокированных ячеек
    blocked = np.zeros((cols, rows), dtype=np.bool_)
    half_thickness = config.WALL_THICKNESS / 2.0
    cell_radius = (cell_w + cell_h) / 4.0  # Радиус ячейки для мягкого определения коллизий

    # Растеризуем стены на сетку
    for col in range(cols):
        cx = (col + 0.5) * cell_w
        for row in range(rows):
            cy = (row + 0.5) * cell_h
            for wall in obstacles:
                x1, y1, x2, y2 = wall
                # Находим кратчайшее расстояние от центра ячейки до отрезка стены
                dx, dy = x2 - x1, y2 - y1
                if dx == 0 and dy == 0:
                    d_sq = (cx - x1) ** 2 + (cy - y1) ** 2
                else:
                    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
                    t = max(0.0, min(1.0, t))
                    d_sq = (cx - (x1 + t * dx)) ** 2 + (cy - (y1 + t * dy)) ** 2

                if d_sq < (half_thickness + cell_radius) ** 2:
                    blocked[col, row] = True
                    break

    # Целевая ячейка всегда должна быть доступна для старта волны
    blocked[goal_col, goal_row] = False

    # Очередь для BFS: (col, row, текущее_расстояние)
    queue = deque([(goal_col, goal_row, 0.0)])
    dist_grid[goal_col, goal_row] = 0.0

    # Направления движения (включая диагонали с соответствующими весами)
    directions = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
    ]

    while queue:
        c, r, d = queue.popleft()

        if d > dist_grid[c, r]:
            continue

        for dc, dr, weight in directions:
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows:
                if not blocked[nc, nr]:
                    new_d = d + weight
                    if new_d < dist_grid[nc, nr]:
                        dist_grid[nc, nr] = new_d
                        queue.append((nc, nr, new_d))

    return dist_grid, cell_w, cell_h


class Population:
    """Управляет жизненным циклом и поведением роя пчел."""

    def __init__(self, brains=None):
        self.pos = np.zeros((config.POP_SIZE, 2), dtype=np.float32)
        # Спавн строго по горизонтальному центру монитора
        self.pos[:] = [config.SCREEN_SIZE[0] // 2, config.SCREEN_SIZE[1] - config.SPAWN_OFFSET_Y]
        self.vel = np.zeros((config.POP_SIZE, 2), dtype=np.float32)
        self.dead = np.zeros(config.POP_SIZE, dtype=np.bool_)
        self.reached = np.zeros(config.POP_SIZE, dtype=np.bool_)
        self.step = 0
        self.is_best_flag = np.zeros(config.POP_SIZE, dtype=np.bool_)

        if brains is not None:
            self.brains = brains
            self.is_best_flag[0] = True
        else:
            self.brains = np.random.uniform(-config.STEP_SIZE, config.STEP_SIZE,
                                            (config.POP_SIZE, config.LIFESPAN, 2)).astype(np.float32)

    def calculate_fitness(self, goal, obstacles=None):
        """Оценка приспособленности пчелы с учетом лабиринтов."""
        screen_w, screen_h = config.SCREEN_SIZE

        # Если препятствий нет, используем стандартное евклидово расстояние
        if obstacles is None or len(obstacles) == 0:
            dist_sq = (self.pos[:, 0] - goal[0]) ** 2 + (self.pos[:, 1] - goal[1]) ** 2
        else:
            grid_size = (50, 50)  # Оптимальный размер сетки для баланса точности и скорости
            dist_grid, cell_w, cell_h = compute_grid_distances(
                obstacles, goal, screen_w, screen_h, grid_size
            )

            pixel_scale = (cell_w + cell_h) / 2.0
            max_possible_dist = (grid_size[0] + grid_size[1]) * pixel_scale
            dist_sq = np.zeros(config.POP_SIZE, dtype=np.float32)

            for i in range(config.POP_SIZE):
                px, py = self.pos[i, 0], self.pos[i, 1]

                # Координаты пчелы на сетке
                col = int(px / cell_w)
                row = int(py / cell_h)
                col = max(0, min(grid_size[0] - 1, col))
                row = max(0, min(grid_size[1] - 1, row))

                grid_steps = dist_grid[col, row]
                euclidean_dist = np.sqrt((px - goal[0]) ** 2 + (py - goal[1]) ** 2)

                if np.isinf(grid_steps):
                    # Если пчела застряла в тупике или в стене, даем ей штрафное расстояние:
                    # евклидово расстояние до цели плюс двойной размер карты
                    actual_dist = euclidean_dist + max_possible_dist * 2.0
                else:
                    # Переводим шаги сетки в пиксели и комбинируем с евклидовым расстоянием,
                    # чтобы у пчел сохранялся плавный градиент движения внутри одной ячейки
                    path_dist_pixels = grid_steps * pixel_scale
                    actual_dist = 0.9 * path_dist_pixels + 0.1 * euclidean_dist

                dist_sq[i] = actual_dist ** 2

        speed_divider = (config.SPEED_BONUS_DIVIDER * self.step / config.LIFESPAN) ** 2
        fitness = 1.0 / (dist_sq * speed_divider + config.FITNESS_BASE_DIVIDER)

        for i in range(config.POP_SIZE):
            if self.reached[i]:
                fitness[i] += config.SUCCESS_BONUS / speed_divider
        return fitness