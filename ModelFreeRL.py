import numpy as np
import random

class ModelFreeAgent:
    def __init__(self, grid, rewards, p, r, discount_factor=0.5, learning_rate=0.1, epsilon=0.1):
        self.h, self.w = len(grid), len(grid[0])
        self.rewards = rewards
        self.p = p
        self.r = r
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q_table = np.zeros((self.h, self.w, 4))  # 4 possible actions: up, down, left, right

        # Convert rewards into a 2D array
        self.reward_grid = np.full((self.h, self.w), r)
        for x, y, reward in self.rewards:
            self.reward_grid[y, x] = reward

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(4))  # Explore
        else:
            return np.argmax(self.q_table[state[1], state[0]])  # Exploit

    def learn(self, episodes=1000):
        for _ in range(episodes):
            state = (np.random.randint(0, self.w), np.random.randint(0, self.h))
            while self.reward_grid[state[1], state[0]] == self.r:  # Non-terminal states
                action = self.choose_action(state)
                new_state = (state[0] + self.actions[action][1], state[1] + self.actions[action][0])
                if 0 <= new_state[0] < self.w and 0 <= new_state[1] < self.h:
                    reward = self.reward_grid[new_state[1], new_state[0]]
                    best_next_action = np.argmax(self.q_table[new_state[1], new_state[0]])
                    td_target = reward + self.discount_factor * self.q_table[new_state[1], new_state[0], best_next_action]
                    td_delta = td_target - self.q_table[state[1], state[0], action]
                    self.q_table[state[1], state[0], action] += self.learning_rate * td_delta
                    state = new_state
                else:
                    break

    def get_policy(self):
        policy = np.zeros((self.h, self.w, 4))
        for y in range(self.h):
            for x in range(self.w):
                best_action = np.argmax(self.q_table[y, x])
                policy[y, x] = np.eye(4)[best_action]
        return policy

    def get_q_table(self):
        return self.q_table
