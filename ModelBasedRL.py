import numpy as np

class ModelBasedAgent:
    def __init__(self, grid, rewards, p, r, discount_factor=0.5):
        self.h, self.w = len(grid), len(grid[0])
        self.rewards = rewards
        self.p = p
        self.r = r
        self.discount_factor = discount_factor
        self.value_function = np.zeros((self.h, self.w))
        self.policy = np.zeros((self.h, self.w, 4))  # 4 possible actions: up, down, left, right

        # Convert rewards into a 2D array
        self.reward_grid = np.full((self.h, self.w), r)
        for x, y, reward in self.rewards:
            self.reward_grid[y, x] = reward

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def learn(self, iterations=1000):
        for _ in range(iterations):
            new_value_function = np.copy(self.value_function)
            for y in range(self.h):
                for x in range(self.w):
                    if (x, y) in [(pos[0], pos[1]) for pos in self.rewards]:  # Skip terminal states
                        continue
                    values = []
                    for action in self.actions:
                        new_y, new_x = y + action[0], x + action[1]
                        if 0 <= new_y < self.h and 0 <= new_x < self.w:
                            values.append(self.p * (self.reward_grid[new_y, new_x] + self.discount_factor * self.value_function[new_y, new_x]) + 
                                          (1 - self.p) / 2 * (self.reward_grid[y, x] + self.discount_factor * self.value_function[y, x]))
                        else:
                            values.append(self.reward_grid[y, x] + self.discount_factor * self.value_function[y, x])
                    new_value_function[y, x] = max(values)
            self.value_function = new_value_function

        for y in range(self.h):
            for x in range(self.w):
                if (x, y) in [(pos[0], pos[1]) for pos in self.rewards]:
                    continue
                values = []
                for action in self.actions:
                    new_y, new_x = y + action[0], x + action[1]
                    if 0 <= new_y < self.h and 0 <= new_x < self.w:
                        values.append(self.p * (self.reward_grid[new_y, new_x] + self.discount_factor * self.value_function[new_y, new_x]) + 
                                      (1 - self.p) / 2 * (self.reward_grid[y, x] + self.discount_factor * self.value_function[y, x]))
                best_action = np.argmax(values)
                self.policy[y, x] = np.eye(4)[best_action]

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.value_function
