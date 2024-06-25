import numpy as np
from GridWorldIterator import *

class ModelBasedAgent:
    def __init__(self, w, h, rewards, p, r, discount_factor=0.5):
        self.h, self.w = h, w
        self.rewards = rewards
        self.p = p
        self.r = r
        self.discount_factor = discount_factor
        self.value_function = np.zeros((self.h, self.w))
        self.policy = np.full((self.h, self.w), '', dtype=object)  # Initialize with empty strings

        # Convert rewards into a 2D array
        self.reward_grid = np.full((self.h, self.w), r)
        for x, y, reward in self.rewards:
            self.reward_grid[self.h-y-1, x] = reward
        self.max_reward = max(pos[2] for pos in rewards)

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        self.action_str = ['up', 'down', 'left', 'right']  # corresponding action strings

    def learn(self, iterations=10000):
        for _ in range(iterations):
            new_value_function = np.copy(self.value_function)
            for y in range(self.h):
                for x in range(self.w):
                    if (x, y) in [pos[:2] for pos in self.rewards]:
                        val = self.reward_grid[h-1-y, x]
                        if self.max_reward == val:
                            self.policy[h-y-1,x]='maxreward' 
                        else: 
                            self.policy[h-y-1,x]='reward' if val>0  else  'negreward' if val < 0  else 'unreward'
                            continue
                    values = []
                    for action in self.actions:
                        new_y, new_x = self.h-1-y + action[0], x + action[1]
                        if 0 <= new_y < self.h and 0 <= new_x < self.w:
                            values.append(self.p * (self.reward_grid[new_y, new_x] + self.discount_factor * self.value_function[new_y, new_x]) + 
                                          (1 - self.p) / 2 * (self.reward_grid[self.h-1-y, x] + self.discount_factor * self.value_function[self.h-1-y, x]))
                        else:
                            values.append(self.reward_grid[self.h-1-y, x] + self.discount_factor * self.value_function[self.h-1-y, x])
                    new_value_function[self.h-1-y, x] = max(values)
            self.value_function = new_value_function

        for y in range(self.h):
            for x in range(self.w):
                if (x, y) in [pos[:2] for pos in self.rewards]:
                    val = self.reward_grid[h-1-y, x]
                    if self.max_reward == val:
                        self.policy[h-y-1,x]='maxreward' 
                    else: 
                        self.policy[h-y-1,x]='reward' if val>0  else  'negreward' if val < 0  else 'unreward'
                        continue
                values = []
                for action in self.actions:
                    new_y, new_x = self.h-1-y + action[0], x + action[1]
                    if 0 <= new_y < self.h and 0 <= new_x < self.w:
                        values.append(self.p * (self.reward_grid[new_y, new_x] + self.discount_factor * self.value_function[new_y, new_x]) + 
                                      (1 - self.p) / 2 * (self.reward_grid[self.h-1-y, x] + self.discount_factor * self.value_function[self.h-1-y, x]))
                best_action = np.argmax(values)
                self.policy[h-1-y, x] = self.action_str[best_action] if self.reward_grid[h-1-y, x]!=self.max_reward else 'maxreward' 

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.value_function

if __name__ == "__main__":
    iterator = GridWorldIterator('GridWorld.py')
    for w, h, L, p, r in iterator:
        md = ModelBasedAgent(w, h, L, p, r)
        md.learn()
        print("Policy:\n", md.get_policy())
        print("Value Function:\n", md.get_value_function())
        break
