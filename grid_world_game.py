import pygame
import numpy as np
from GridWorldIterator import *


def draw_grid(screen, grid, cell_size, rewards):
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
            if (x, y) in [(pos[0], pos[1]) for pos in rewards]:
                color = (0, 255, 0) if grid[y][x] > 0 else (255, 0, 0)
                pygame.draw.rect(screen, color, rect)
            text = font.render(f'{grid[y][x]}', True, (0, 0, 0))
            screen.blit(text, (x * cell_size + 10, y * cell_size + 10))

def draw_arrow(screen, pos, direction, cell_size):
    center_x = pos[0] * cell_size + cell_size // 2
    center_y = pos[1] * cell_size + cell_size // 2
    arrow_length = cell_size // 4  # Half of the distance to the edge
    arrow_head_length = 10  # Length of the arrow head

    if direction == "up":
        pygame.draw.line(screen, (0, 0, 0), (center_x, center_y + arrow_length), (center_x, center_y - arrow_length), 5)
        pygame.draw.polygon(screen, (0, 0, 0), [(center_x, center_y - arrow_length), 
                                                (center_x - arrow_head_length, center_y - arrow_length + arrow_head_length), 
                                                (center_x + arrow_head_length, center_y - arrow_length + arrow_head_length)])
    elif direction == "down":
        pygame.draw.line(screen, (0, 0, 0), (center_x, center_y - arrow_length), (center_x, center_y + arrow_length), 5)
        pygame.draw.polygon(screen, (0, 0, 0), [(center_x, center_y + arrow_length), 
                                                (center_x - arrow_head_length, center_y + arrow_length - arrow_head_length), 
                                                (center_x + arrow_head_length, center_y + arrow_length - arrow_head_length)])
    elif direction == "left":
        pygame.draw.line(screen, (0, 0, 0), (center_x + arrow_length, center_y), (center_x - arrow_length, center_y), 5)
        pygame.draw.polygon(screen, (0, 0, 0), [(center_x - arrow_length, center_y), 
                                                (center_x - arrow_length + arrow_head_length, center_y - arrow_head_length), 
                                                (center_x - arrow_length + arrow_head_length, center_y + arrow_head_length)])
    elif direction == "right":
        pygame.draw.line(screen, (0, 0, 0), (center_x - arrow_length, center_y), (center_x + arrow_length, center_y), 5)
        pygame.draw.polygon(screen, (0, 0, 0), [(center_x + arrow_length, center_y), 
                                                (center_x + arrow_length - arrow_head_length, center_y - arrow_head_length), 
                                                (center_x + arrow_length - arrow_head_length, center_y + arrow_head_length)])

def move_player(player_pos, direction, width, height):
    if direction == "up":
        return (player_pos[0], max(0, player_pos[1] - 1))
    elif direction == "down":
        return (player_pos[0], min(height - 1, player_pos[1] + 1))
    elif direction == "left":
        return (max(0, player_pos[0] - 1), player_pos[1])
    elif direction == "right":
        return (min(width - 1, player_pos[0] + 1), player_pos[1])
    return player_pos

# אתחול pygame
pygame.init()
filename = 'GridWorld.py'
iterator = GridWorldIterator(filename)

# קבלת הפרמטרים הראשונים
try:
    w, h, L, p, r = next(iterator)
except StopIteration:
    print("No grids found in file.")
    exit()

GRID_WIDTH = w
GRID_HEIGHT = h
CELL_SIZE = 100
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Grid World Game")
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

# יצירת מפת פרסים ועונשים
reward_grid = np.full((h, w), r)
for x, y, reward in L:
    reward_grid[y, x] = reward

# מיקום התחלתי של השחקן
player_pos = (0, 0)
direction = None

# רשימת מיקומים וכיוונים
arrows = {}

# לולאת המשחק
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                direction = "up"
                player_pos = move_player(player_pos, "up", w, h)
            elif event.key == pygame.K_DOWN:
                direction = "down"
                player_pos = move_player(player_pos, "down", w, h)
            elif event.key == pygame.K_LEFT:
                direction = "left"
                player_pos = move_player(player_pos, "left", w, h)
            elif event.key == pygame.K_RIGHT:
                direction = "right"
                player_pos = move_player(player_pos, "right", w, h)
            elif event.key == pygame.K_RETURN:  # מקש אנטר להחלפת גריד
                try:
                    w, h, L, p, r = next(iterator)
                    GRID_WIDTH = w
                    GRID_HEIGHT = h
                    SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
                    SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
                    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                    reward_grid = np.full((h, w), r)
                    for x, y, reward in L:
                        reward_grid[y, x] = reward
                    player_pos = (0, 0)
                    direction = None
                    arrows.clear()
                except StopIteration:
                    print("No more grids available.")
                    continue

            # שמירת הכיוון עבור המיקום הנוכחי
            arrows[player_pos] = direction

    screen.fill((255, 255, 255))
    draw_grid(screen, reward_grid, CELL_SIZE, L)

    # ציור השחקן
    player_rect = pygame.Rect(player_pos[0] * CELL_SIZE, player_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (0, 0, 255), player_rect)
    
    # ציור החצים
    for pos, direction in arrows.items():
        draw_arrow(screen, pos, direction, CELL_SIZE)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
