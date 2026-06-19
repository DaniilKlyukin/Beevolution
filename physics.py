import random
import numpy as np
from numba import njit


@njit
def dist_to_segment_sq(px, py, x1, y1, x2, y2):
    """Математика: находит кратчайшее расстояние от точки до отрезка линии."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return (px - x1) ** 2 + (py - y1) ** 2
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    return (px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2


@njit
def update_physics_jit(pos, vel, brains, step, dead, reached, obstacles, goal, rad_sq, wall_th, push_f, max_s, screen_w,
                       screen_h):
    """Основной физический цикл обновления координат с ограничением по экрану."""
    half_th_sq = (wall_th / 2) ** 2
    for i in range(len(pos)):
        if not dead[i] and not reached[i]:
            acc_x, acc_y = brains[i, step, 0], brains[i, step, 1]
            vel[i, 0] += acc_x
            vel[i, 1] += acc_y

            speed = (vel[i, 0] ** 2 + vel[i, 1] ** 2) ** 0.5
            if speed > max_s:
                vel[i, 0] = (vel[i, 0] / speed) * max_s
                vel[i, 1] = (vel[i, 1] / speed) * max_s

            pos[i, 0] += vel[i, 0]
            pos[i, 1] += vel[i, 1]

            # Ограничение по границам экрана
            if pos[i, 0] < 0 or pos[i, 0] > screen_w or pos[i, 1] < 0 or pos[i, 1] > screen_h:
                dead[i] = True

            for j in range(len(obstacles)):
                x1, y1, x2, y2 = obstacles[j]
                if dist_to_segment_sq(pos[i, 0], pos[i, 1], x1, y1, x2, y2) < half_th_sq:
                    obstacles[j, 0] += vel[i, 0] * push_f
                    obstacles[j, 1] += vel[i, 1] * push_f
                    obstacles[j, 2] += vel[i, 0] * push_f
                    obstacles[j, 3] += vel[i, 1] * push_f
                    dead[i] = True

            if (pos[i, 0] - goal[0]) ** 2 + (pos[i, 1] - goal[1]) ** 2 < rad_sq:
                reached[i] = True


@njit
def evolve_jit(brains, fitness, mutation_rate, step_s, elite_p):
    """Селекция, кроссинговер и мутация генома."""
    size, lifespan = brains.shape[0], brains.shape[1]
    idx = np.argsort(fitness)[::-1]
    mating_pool = idx[:int(size * elite_p)]
    new_brains = np.zeros_like(brains)

    new_brains[0] = brains[idx[0]]

    for i in range(1, size):
        p1 = mating_pool[random.randint(0, len(mating_pool) - 1)]
        p2 = mating_pool[random.randint(0, len(mating_pool) - 1)]

        mid = random.randint(0, lifespan - 1)
        for s in range(lifespan):
            if s < mid:
                new_brains[i, s] = brains[p1, s]
            else:
                new_brains[i, s] = brains[p2, s]

            if random.random() < mutation_rate:
                new_brains[i, s, 0] = random.uniform(-step_s, step_s)
                new_brains[i, s, 1] = random.uniform(-step_s, step_s)
    return new_brains