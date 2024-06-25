import pygame
import numpy as np
from GridWorldIterator import *
from settings import *
from ValueIteration import *
from ModelBasedRL import *
from ModelFreeRL import *

class Game:
    
    def __init__(self, file_name: str) -> None:
        pygame.init()
        pygame.display.set_caption("Grid World Game")
        self.font = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()
        self.iterator = GridWorldIterator(file_name)
        self.init()
     
    def init(self):
        try:
            self.w, self.h, self.L, self.p, self.r = next(self.iterator)
            self.screen = pygame.display.set_mode((self.w * CELL_SIZE, self.h * CELL_SIZE + 50))
            self.reward_grid = np.full((self.h, self.w), self.r)
            for x, y, reward in self.L:
                self.reward_grid[self.h-1-y, x] = reward
            self.max_reward = max(pos[2] for pos in self.L)
            self.player_pos = (0, self.h-1)
            self.direction = None
            self.arrows = {}

            # Button definitions
            w = (self.w * CELL_SIZE)//4
            self.buttons = {
                "IV": pygame.Rect(0, self.h * CELL_SIZE + 10, w, 30),
                "MB": pygame.Rect(w, self.h * CELL_SIZE + 10, w, 30),
                "CLS": pygame.Rect(2*w, self.h * CELL_SIZE + 10, w, 30),
                "MF": pygame.Rect(3*w, self.h * CELL_SIZE + 10, w, 30)
            }
        except StopIteration:
            print("No more grids available.")
  
    def draw_agent(self):
        player_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        player_surface.fill((0, 0, 255, 128))  # 128 is the alpha value for 50% opacity
        player_rect = pygame.Rect(self.player_pos[0] * CELL_SIZE, self.player_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        self.screen.blit(player_surface, player_rect.topleft)
        
    def draw_grid(self):
        screen = self.screen
        for y in range(len(self.reward_grid)):
            for x in range(len(self.reward_grid[0])):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (200, 200, 200), rect, 1)
                if (x, self.h-y-1) in [(pos[0], pos[1]) for pos in self.L]:
                    color = (0, 255, 0) if self.reward_grid[y][x] ==self.max_reward else (255, 0, 0) if 0> self.reward_grid[y][x] else (0, 0, 255)
                    pygame.draw.rect(screen, color, rect)
                text = self.font.render(f'{self.reward_grid[y][x]}', True, (0, 0, 0))
                screen.blit(text, (x * CELL_SIZE + 10, y * CELL_SIZE + 10))

    def draw_arrow(self):
        screen = self.screen
        for pos, direction in self.arrows.items():
            center_x = pos[0] * CELL_SIZE + CELL_SIZE // 2
            center_y = pos[1] * CELL_SIZE + CELL_SIZE // 2
            arrow_length = (CELL_SIZE // 4) * 0.7  # 30% smaller
            arrow_head_length = 10  # 30% smaller

            if direction == "up":
                pygame.draw.line(screen, (0, 0, 0), (center_x, center_y + arrow_length), (center_x, center_y - arrow_length + 2), 5)
                pygame.draw.polygon(screen, (0, 0, 0), [(center_x, center_y - arrow_length), 
                                                        (center_x - arrow_head_length, center_y - arrow_length + arrow_head_length), 
                                                        (center_x + arrow_head_length, center_y - arrow_length + arrow_head_length)])
            elif direction == "down":
                pygame.draw.line(screen, (0, 0, 0), (center_x, center_y - arrow_length + 2), (center_x, center_y + arrow_length), 5)
                pygame.draw.polygon(screen, (0, 0, 0), [(center_x, center_y + arrow_length), 
                                                        (center_x - arrow_head_length, center_y + arrow_length - arrow_head_length), 
                                                        (center_x + arrow_head_length, center_y + arrow_length - arrow_head_length)])
            elif direction == "left":
                pygame.draw.line(screen, (0, 0, 0), (center_x + arrow_length - 2, center_y), (center_x - arrow_length + 2, center_y), 5)
                pygame.draw.polygon(screen, (0, 0, 0), [(center_x - arrow_length, center_y), 
                                                        (center_x - arrow_length + arrow_head_length, center_y - arrow_head_length), 
                                                        (center_x - arrow_length + arrow_head_length, center_y + arrow_head_length)])
            elif direction == "right":
                pygame.draw.line(screen, (0, 0, 0), (center_x - arrow_length, center_y), (center_x + arrow_length - 2, center_y), 5)
                pygame.draw.polygon(screen, (0, 0, 0), [(center_x + arrow_length, center_y), 
                                                        (center_x + arrow_length - arrow_head_length, center_y - arrow_head_length), 
                                                        (center_x + arrow_length - arrow_head_length, center_y + arrow_head_length)])
                
            else:   
                rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE,CELL_SIZE,CELL_SIZE)        
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
                text = self.font.render(direction, True, (0, 0, 0))
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)

            

    def move_player(self, event):
        player_pos = self.player_pos
        if event.key == pygame.K_UP:
            self.direction = "up"
            self.player_pos = (player_pos[0], max(0, player_pos[1] - 1))
        elif event.key == pygame.K_DOWN:
            self.direction = "down"
            self.player_pos = (player_pos[0], min(self.h - 1, player_pos[1] + 1))
        elif event.key == pygame.K_LEFT:
            self.direction = "left"
            self.player_pos = (max(0, player_pos[0] - 1), player_pos[1])
        elif event.key == pygame.K_RIGHT:
            self.direction = "right"
            self.player_pos = (min(self.w - 1, player_pos[0] + 1), player_pos[1])
        
        self.arrows[self.player_pos] = self.direction 
        

    def button_iv(self):
        policy, value_function = value_iteration(self.w, self.h, self.L, self.p, self.r)
        for i in range(policy.shape[1]):
            for j in range(policy.shape[0]):
                self.arrows[(i, j)] = policy[j, i]
        print("Policy:\n", policy)
        print("Value Function:\n", value_function)
        
    
    def button_mb(self):
        policy, value_function = value_iteration(self.w, self.h, self.L, self.p, self.r)
        for i in range(policy.shape[1]):
            for j in range(policy.shape[0]):
                self.arrows[(i, j)] = policy[j, i]
        print("Policy:\n", policy)
        print("Value Function:\n", value_function)
        
    
    def button_cls(self):
        self.arrows.clear()
        self.player_pos = (0, self.h-1)
        
    
    def button_mf(self):
        policy, value_function = value_iteration(self.w, self.h, self.L, self.p, self.r)
        for i in range(policy.shape[1]):
            for j in range(policy.shape[0]):
                self.arrows[(i, j)] = policy[j, i]
        print("Policy:\n", policy)
        print("Value Function:\n", value_function)
        
    
    def get_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                self.move_player(event)
                if event.key == pygame.K_RETURN:
                    self.init()
                    self.screen
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = event.pos
                    if self.buttons["IV"].collidepoint(mouse_pos):
                        self.button_iv()
                    elif self.buttons["MB"].collidepoint(mouse_pos):
                        self.button_mb()
                    elif self.buttons["CLS"].collidepoint(mouse_pos):
                        self.button_cls()
                    elif self.buttons["MF"].collidepoint(mouse_pos):
                        self.button_mf()
        return True

    def draw_buttons(self):
        for label, rect in self.buttons.items():
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
            text = self.font.render(label, True, (0, 0, 0))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def __call__(self) -> None:
        running = True
        while running:
            running = self.get_events()
            self.screen.fill((255, 255, 255))
            self.draw_grid()
            self.draw_arrow()
            self.draw_agent()
            self.draw_buttons()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    game = Game('GridWorld.py')
    game()
