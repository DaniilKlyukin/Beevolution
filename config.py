import pygame

pygame.init()
info = pygame.display.Info()

# Определяем нативное разрешение монитора (с безопасным резервным вариантом)
monitor_w = info.current_w if info.current_w > 0 else 1920
monitor_h = info.current_h if info.current_h > 0 else 1080

SCREEN_SIZE = (monitor_w, monitor_h)
FPS_LIMIT = 60  # Восстановленный параметр ограничения кадров в секунду

# Квадратная игровая зона по высоте экрана
PLAYABLE_SIDE = monitor_h
LEFT_MARGIN = (monitor_w - PLAYABLE_SIDE) // 2

# Расчет высоты элементов относительно экрана
GOAL_Y = int(0.08 * PLAYABLE_SIDE)
SPAWN_OFFSET_Y = int(0.07 * PLAYABLE_SIDE)

# Физические параметры
MAX_SPEED = 7.0
WALL_THICKNESS = 12
PUSH_FORCE = 0.15
MIN_WALL_LENGTH = 5

# Популяция и ГА
POP_SIZE = 200
LIFESPAN = 500
STEP_SIZE = 0.7
MUTATION_MULTIPLIER = 2
START_MUTATION = 0.01
ELITE_PERCENTAGE = 0.2

# Фитнес
FITNESS_BASE_DIVIDER = 1.0
SUCCESS_BONUS = 1000.0
SPEED_BONUS_DIVIDER = 5000.0

# Визуальные параметры
COLOR_BG = (15, 15, 25)
COLOR_BEE_YELLOW = (255, 200, 50)
COLOR_BEE_BLACK = (30, 30, 30)
COLOR_BEE_BEST = (0, 255, 150)
COLOR_BEE_HIT = (255, 100, 100)
COLOR_BEE_STUNG = (100, 255, 100)
COLOR_WALL = (120, 130, 160)
COLOR_WALL_DRAWING = (200, 210, 230, 150)
COLOR_BORDER = (80, 90, 110)  # Цвет рамок игровой зоны
COLOR_TEXT = (240, 240, 250)
COLOR_HITS = (255, 215, 0)
COLOR_INFO = (130, 200, 250)
COLOR_TRAIL = (0, 255, 150, 70)

BEE_RADIUS = 6
BEE_RADIUS_BEST = 8
STRIPE_WIDTH = 2

GOAL_WIDTH = int(0.10 * PLAYABLE_SIDE)
GOAL_HEIGHT = int(0.10 * PLAYABLE_SIDE)
COLLISION_RADIUS_SQ = (min(GOAL_WIDTH, GOAL_HEIGHT) // 2) ** 2