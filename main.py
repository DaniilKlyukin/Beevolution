import pygame
import numpy as np
import math
import sys
import config
from physics import update_physics_jit, evolve_jit
from population import Population
from ui import draw_bee, draw_hud, TUTORIAL_STEPS
from presets import get_preset_obstacles  # Импорт пресетов


def compose_background(loaded_bg, screen_w, screen_h, playable_side, left_margin):
    """
    Адаптивно собирает панорамный зеркальный фон под любое разрешение монитора.
    Использует метод перекрытия (overdraw) для гарантированного устранения видимых швов.
    """
    core_bg = pygame.transform.scale(loaded_bg, (playable_side, playable_side))
    full_bg = pygame.Surface((screen_w, screen_h))

    if left_margin > 0:
        slice_width = min(left_margin, playable_side)

        # --- ЛЕВАЯ ОБЛАСТЬ ---
        left_slice = core_bg.subsurface((0, 0, slice_width, playable_side))
        left_flipped = pygame.transform.flip(left_slice, True, False)
        left_flipped_scaled = pygame.transform.scale(left_flipped, (left_margin + 2, playable_side))
        full_bg.blit(left_flipped_scaled, (0, 0))

        # --- ПРАВАЯ ОБЛАСТЬ ---
        right_slice = core_bg.subsurface((playable_side - slice_width, 0, slice_width, playable_side))
        right_flipped = pygame.transform.flip(right_slice, True, False)
        right_flipped_scaled = pygame.transform.scale(right_flipped, (left_margin + 2, playable_side))
        full_bg.blit(right_flipped_scaled, (left_margin + playable_side - 2, 0))

    # --- ЦЕНТРАЛЬНЫЙ КВАДРАТ ---
    full_bg.blit(core_bg, (left_margin, 0))

    return full_bg


def main():
    pygame.init()
    # Запуск в полноценном fullscreen-режиме
    screen = pygame.display.set_mode(config.SCREEN_SIZE, pygame.FULLSCREEN)
    pygame.display.set_caption("Эволюция пчел: Генетический Алгоритм")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Consolas", 16, bold=True)
    font_large = pygame.font.SysFont("Consolas", 20, bold=True)
    font_title = pygame.font.SysFont("Consolas", 24, bold=True)

    # 1. Загрузка текстур
    try:
        img_goal = pygame.image.load("assets/goal.png").convert_alpha()
        img_hit = pygame.image.load("assets/hit.png").convert_alpha()
        img_goal = pygame.transform.scale(img_goal, (config.GOAL_WIDTH, config.GOAL_HEIGHT))
        img_hit = pygame.transform.scale(img_hit, (config.GOAL_WIDTH, config.GOAL_HEIGHT))
    except Exception:
        img_goal = pygame.Surface((config.GOAL_WIDTH, config.GOAL_HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(img_goal, (255, 180, 50), (config.GOAL_WIDTH // 2, config.GOAL_HEIGHT // 2),
                           config.GOAL_WIDTH // 2)
        img_hit = pygame.Surface((config.GOAL_WIDTH, config.GOAL_HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(img_hit, (255, 100, 100), (config.GOAL_WIDTH // 2, config.GOAL_HEIGHT // 2),
                           config.GOAL_WIDTH // 2)

    try:
        raw_background = pygame.image.load("assets/background.png").convert()
        background = compose_background(raw_background, config.SCREEN_SIZE[0], config.SCREEN_SIZE[1],
                                        config.PLAYABLE_SIDE, config.LEFT_MARGIN)
    except Exception:
        background = pygame.Surface(config.SCREEN_SIZE)
        background.fill(config.COLOR_BG)

    # 2. Инициализация переменных физики и роя
    goal_pos = np.array([config.SCREEN_SIZE[0] // 2, config.GOAL_Y], dtype=np.float32)
    img_rect = img_goal.get_rect(center=(int(goal_pos[0]), int(goal_pos[1])))

    original_obstacles = np.zeros((0, 4), dtype=np.float32)
    obstacles = np.zeros((0, 4), dtype=np.float32)

    pop = Population()
    generation = 1
    mutation_rate = config.START_MUTATION
    best_gen_hits = 0
    current_gen_hits = 0
    drawing_wall = False
    start_pos = (0, 0)

    leader_trail = []
    saved_leader_trail = []
    trail_surface = pygame.Surface(config.SCREEN_SIZE, pygame.SRCALPHA)

    # Переменные интерактивного обучения
    tutorial_active = True
    tutorial_step = 0
    step_2_running = False

    # 3. Объявление вложенных функций в правильном порядке
    def next_generation():
        """Смена поколений (вызывается в конце жизненного цикла роя)."""
        nonlocal pop, generation, best_gen_hits, obstacles, current_gen_hits, leader_trail, saved_leader_trail, step_2_running, tutorial_step
        current_gen_hits = np.sum(pop.reached)
        best_gen_hits = max(best_gen_hits, current_gen_hits)

        saved_leader_trail = leader_trail.copy()
        leader_trail = []

        pop = Population(
            evolve_jit(pop.brains, pop.calculate_fitness(goal_pos, obstacles), mutation_rate, config.STEP_SIZE,
                       config.ELITE_PERCENTAGE))
        generation += 1
        obstacles = original_obstacles.copy()

        if step_2_running:
            step_2_running = False
            tutorial_step = 3

    def set_preset(preset_type):
        """Загрузка препятствий из файла пресетов."""
        nonlocal original_obstacles, obstacles
        original_obstacles = get_preset_obstacles(
            preset_type, config.SCREEN_SIZE[0], config.SCREEN_SIZE[1], config.PLAYABLE_SIDE, config.LEFT_MARGIN
        )
        obstacles = original_obstacles.copy()

    # 4. Основной игровой цикл
    running = True
    while running:
        m_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if tutorial_active:
                    if event.button == 1:
                        if tutorial_step == 2 and not step_2_running:
                            step_2_running = True
                        elif tutorial_step != 2:
                            if tutorial_step < len(TUTORIAL_STEPS) - 1:
                                tutorial_step += 1
                            else:
                                tutorial_active = False
                else:
                    if event.button == 1:
                        # Разрешаем строить стены в пределах игрового поля
                        if config.LEFT_MARGIN <= m_pos[0] <= config.LEFT_MARGIN + config.PLAYABLE_SIDE:
                            drawing_wall, start_pos = True, m_pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and drawing_wall and not tutorial_active:
                    if math.dist(start_pos, m_pos) > config.MIN_WALL_LENGTH:
                        new_wall = np.array([[start_pos[0], start_pos[1], m_pos[0], m_pos[1]]], dtype=np.float32)
                        original_obstacles = np.append(original_obstacles, new_wall, axis=0)
                        obstacles = original_obstacles.copy()
                    drawing_wall = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if tutorial_active:
                    if event.key in [pygame.K_SPACE, pygame.K_RETURN]:
                        if tutorial_step == 2 and not step_2_running:
                            step_2_running = True
                        elif tutorial_step != 2:
                            if tutorial_step < len(TUTORIAL_STEPS) - 1:
                                tutorial_step += 1
                            else:
                                tutorial_active = False

                    elif event.key in [pygame.K_LEFT, pygame.K_BACKSPACE]:
                        if step_2_running:
                            step_2_running = False
                            pop = Population()
                            leader_trail = []
                        else:
                            if tutorial_step > 0:
                                tutorial_step -= 1
                else:
                    if event.key == pygame.K_UP: mutation_rate = min(1.0, mutation_rate * config.MUTATION_MULTIPLIER)
                    if event.key == pygame.K_DOWN: mutation_rate = max(2 ** -16,
                                                                       mutation_rate / config.MUTATION_MULTIPLIER)
                    if event.key == pygame.K_c: set_preset(4)
                    if event.key == pygame.K_d and len(original_obstacles) > 0:
                        original_obstacles = original_obstacles[:-1]
                        obstacles = original_obstacles.copy()
                    if event.key == pygame.K_r:
                        pop = Population()
                        generation = 1
                        best_gen_hits = 0
                        leader_trail = []
                        saved_leader_trail = []
                        set_preset(4)

                    if event.key == pygame.K_1: set_preset(1)
                    if event.key == pygame.K_2: set_preset(2)
                    if event.key == pygame.K_3: set_preset(3)
                    if event.key == pygame.K_4: set_preset(4)
                    if event.key == pygame.K_5: set_preset(5)

                    fast_gens = 10 if event.key == pygame.K_SPACE else 50 if event.key == pygame.K_f else 200 if event.key == pygame.K_g else 0
                    if fast_gens > 0:
                        for _ in range(fast_gens):
                            while pop.step < config.LIFESPAN and not np.all(pop.dead | pop.reached):
                                update_physics_jit(pop.pos, pop.vel, pop.brains, pop.step, pop.dead, pop.reached,
                                                   obstacles, goal_pos, config.COLLISION_RADIUS_SQ,
                                                   config.WALL_THICKNESS, config.PUSH_FORCE,
                                                   config.MAX_SPEED, config.SCREEN_SIZE[0], config.SCREEN_SIZE[1])
                                pop.step += 1
                            next_generation()

        # --- ОБНОВЛЕНИЕ МИРА ---
        if not tutorial_active or step_2_running:
            if pop.step < config.LIFESPAN and not np.all(pop.dead | pop.reached):
                leader_trail.append((pop.pos[0, 0], pop.pos[0, 1]))
                update_physics_jit(pop.pos, pop.vel, pop.brains, pop.step, pop.dead, pop.reached,
                                   obstacles, goal_pos, config.COLLISION_RADIUS_SQ, config.WALL_THICKNESS,
                                   config.PUSH_FORCE, config.MAX_SPEED,
                                   config.SCREEN_SIZE[0], config.SCREEN_SIZE[1])
                pop.step += 1
            else:
                next_generation()

        # --- ОТРИСОВКА ---
        screen.blit(background, (0, 0))

        # Отрисовка траектории
        if len(saved_leader_trail) > 1:
            trail_surface.fill((0, 0, 0, 0))
            pygame.draw.lines(trail_surface, config.COLOR_TRAIL, False,
                              [(int(p[0]), int(p[1])) for p in saved_leader_trail], 4)
            for idx, pt in enumerate(saved_leader_trail):
                if idx % 40 == 0:
                    pygame.draw.circle(trail_surface, (0, 255, 150, 180), (int(pt[0]), int(pt[1])), 3)
            screen.blit(trail_surface, (0, 0))

        # Дого
        goal_img = img_hit if np.any(pop.reached) else img_goal
        screen.blit(goal_img, img_rect)

        # Стены
        for wall in obstacles:
            pygame.draw.line(screen, config.COLOR_WALL, (int(wall[0]), int(wall[1])), (int(wall[2]), int(wall[3])),
                             config.WALL_THICKNESS)
        if drawing_wall:
            pygame.draw.line(screen, config.COLOR_WALL_DRAWING, start_pos, m_pos, config.WALL_THICKNESS)

        # Пчелы
        for i in range(config.POP_SIZE):
            if not pop.dead[i]:
                draw_bee(screen, pop.pos[i, 0], pop.pos[i, 1], pop.is_best_flag[i], pop.reached[i], pop.dead[i])

        # Боковой HUD
        if not tutorial_active:
            if mutation_rate > 0.08:
                behavior_desc = "Поиск путей (Хаос)"
                behavior_color = (255, 120, 120)
            elif mutation_rate > 0.01:
                behavior_desc = "Оптимальный баланс"
                behavior_color = (255, 255, 150)
            else:
                behavior_desc = "Стабилизация"
                behavior_color = (150, 255, 150)

            draw_hud(screen, pop, generation, best_gen_hits, current_gen_hits, mutation_rate, clock, font,
                     behavior_desc, behavior_color)

        # Слайды обучения
        else:
            if step_2_running:
                # Широкий нижний статус-бар (1050 пикселей)
                hud_panel = pygame.Surface((1050, 50))
                hud_panel.fill((15, 20, 35))
                hud_panel.set_alpha(220)
                screen.blit(hud_panel, ((config.SCREEN_SIZE[0] - 1050) // 2, config.SCREEN_SIZE[1] - 80))
                pygame.draw.rect(screen, (230, 160, 30),
                                 ((config.SCREEN_SIZE[0] - 1050) // 2, config.SCREEN_SIZE[1] - 80, 1050, 50), width=2,
                                 border_radius=5)

                status_text = f"Поколение №1 в полете... Шаг: {pop.step}/{config.LIFESPAN}. Нажмите [Backspace] для отмены."
                status_render = font_large.render(status_text, True, config.COLOR_TEXT)
                screen.blit(status_render,
                            ((config.SCREEN_SIZE[0] - status_render.get_width()) // 2, config.SCREEN_SIZE[1] - 68))
            else:
                current_slide = TUTORIAL_STEPS[tutorial_step]
                darken_overlay = pygame.Surface(config.SCREEN_SIZE, pygame.SRCALPHA)
                darken_overlay.fill((10, 10, 20, 180))
                screen.blit(darken_overlay, (0, 0))

                hl = current_slide["highlight"]
                if hl == "endpoints":
                    pygame.draw.circle(screen, (255, 255, 255), (int(goal_pos[0]), int(goal_pos[1])),
                                       int(0.10 * config.PLAYABLE_SIDE), 4)
                    pygame.draw.circle(screen, (255, 255, 255),
                                       (config.SCREEN_SIZE[0] // 2, config.SCREEN_SIZE[1] - config.SPAWN_OFFSET_Y), 60,
                                       4)
                elif hl == "spawn":
                    pygame.draw.circle(screen, (255, 215, 0),
                                       (config.SCREEN_SIZE[0] // 2, config.SCREEN_SIZE[1] - config.SPAWN_OFFSET_Y), 70,
                                       4)
                elif hl == "mutation" and len(saved_leader_trail) > 1:
                    for pt in saved_leader_trail[::50]:
                        pygame.draw.circle(screen, config.COLOR_BEE_BEST, (int(pt[0]), int(pt[1])), 25, 2)

                # Основное окно обучения расширено до 1000px
                box_width = min(1000, config.SCREEN_SIZE[0] - 80)
                box_height = min(410, config.SCREEN_SIZE[1] - 100)
                box_x = (config.SCREEN_SIZE[0] - box_width) // 2
                box_y = (config.SCREEN_SIZE[1] - box_height) // 2

                pygame.draw.rect(screen, (5, 5, 10), (box_x + 8, box_y + 8, box_width, box_height), border_radius=15)
                pygame.draw.rect(screen, (25, 30, 50), (box_x, box_y, box_width, box_height), border_radius=15)
                pygame.draw.rect(screen, (230, 160, 30), (box_x, box_y, box_width, box_height), width=3,
                                 border_radius=15)

                title_render = font_title.render(current_slide["title"], True, config.COLOR_HITS)
                screen.blit(title_render, (box_x + (box_width - title_render.get_width()) // 2, box_y + 25))

                pygame.draw.line(screen, config.COLOR_WALL, (box_x + 40, box_y + 65),
                                 (box_x + box_width - 40, box_y + 65), 2)

                for line_idx, line in enumerate(current_slide["lines"]):
                    text_render = font_large.render(line, True, config.COLOR_TEXT)
                    screen.blit(text_render, (box_x + 50, box_y + 90 + line_idx * 26))

                nav_lines = []
                if tutorial_step > 0:
                    nav_lines.append("[Backspace / Стрелка влево] Назад")
                if tutorial_step == 2:
                    nav_lines.append("[Пробел / Клик] Запуск Поколения №1 »")
                else:
                    nav_lines.append("[Пробел / Клик] Продолжить »")

                nav_text = "  |  ".join(nav_lines)
                action_render = font_large.render(nav_text, True, config.COLOR_INFO)
                screen.blit(action_render,
                            (box_x + (box_width - action_render.get_width()) // 2, box_y + box_height - 35))

        pygame.display.flip()
        clock.tick(config.FPS_LIMIT)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()