import numpy as np
from GridWorldBuilder import GridWorldBuilder
import random

random.seed(42)

class ModelFreeRL:
    """
    Q-Learning agent for solving grid world problems.
    """
    
    def __init__(self, grid: GridWorldBuilder, discount_factor=0.5, epsilon=0.01, decay=0.99, learning_rate=0.01, episodes=1000) -> None:
        """
        Initialize the QLearningAgent with given parameters.

        Args:
            grid (GridWorldBuilder): The grid world builder instance.
            discount_factor (float): The discount factor for future rewards.
            epsilon (float): The probability for exploration in the epsilon-greedy policy.
            decay (float): The decay rate for epsilon.
            learning_rate (float): The learning rate for Q-learning updates.
            episodes (int): The number of training episodes.
        """
        self.grid = grid
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.decay = decay
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.rewards = self.initialize_rewards()
        self.actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        self.q_values = np.zeros((self.grid.num_states, self.grid.num_actions))

    def initialize_rewards(self):
        """
        Initialize the reward function for the grid world.

        Returns:
            list: A list of rewards for each state in the grid world.
        """
        rewards = [(x, self.grid.h-1-y, val) for x, y, val in self.grid.current_grid['reward']]
        reward_grid = np.full((self.grid.h, self.grid.w), -1)
        for x, y, val in rewards:
            reward_grid[y][x] = val
        return rewards

    def epsilon_greedy_policy(self, state):
        """
        Select an action using the epsilon-greedy policy.

        Args:
            state (int): The current state index.

        Returns:
            int: The selected action.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.grid.num_actions - 1)
        else:
            return np.argmax(self.q_values[state])

    def get_next_state(self, state, action):
        """
        Get the next state based on the current state and action.

        Args:
            state (int): The current state index.
            action (int): The action taken.

        Returns:
            int: The next state index.
        """
        r, c = self.grid.get_pos_from_state(state)
        dr, dc = self.actions[action]
        new_r, new_c = r + dr, c + dc
        if new_r < 0 or new_c < 0 or new_r >= self.grid.h or new_c >= self.grid.w or ((new_r, new_c) in [(py, px) for px, py, val in self.rewards if val == 0]):
            return state
        else:
            return self.grid.get_state_from_pos((new_r, new_c))

    def train(self):
        """
        Train the Q-learning agent using the grid world.
        """
        epsilon = self.epsilon
        
        for episode in range(self.episodes):
            state = random.randint(0, self.grid.num_states - 1)
            while state in [self.grid.get_state_from_pos((py, px)) for px, py, val in self.rewards if val != -1]:
                state = random.randint(0, self.grid.num_states - 1)
            while state not in [self.grid.get_state_from_pos((py, px)) for px, py, val in self.rewards if val != -1]:
                action = self.epsilon_greedy_policy(state)
                next_state = self.get_next_state(state, action)
                reward = self.grid.reward_table[next_state]
                best_next_action = np.max(self.q_values[next_state])
                self.q_values[state][action] += self.learning_rate * (reward + self.discount_factor * best_next_action - self.q_values[state][action])
                state = next_state
            epsilon = max(0.01, epsilon * self.decay)

    def get_policy(self):
        """
        Extract the optimal policy from the Q-values.

        Returns:
            np.ndarray: The optimal policy.
        """
        policy = np.zeros((self.grid.h, self.grid.w), dtype=int)
        for r in range(self.grid.h):
            for c in range(self.grid.w):
                state = self.grid.get_state_from_pos((r, c))
                policy[r][c] = np.argmax(self.q_values[state])
        return policy

    def get_values(self):
        """
        Extract the state values from the Q-values.

        Returns:
            np.ndarray: The state values.
        """
        values = np.zeros((self.grid.h, self.grid.w))
        for r in range(self.grid.h):
            for c in range(self.grid.w):
                state = self.grid.get_state_from_pos((r, c))
                values[r][c] = np.max(self.q_values[state])
        return values
    def get_values_(self):
        """
        Extract the state values from the Q-values.

        Returns:
            np.ndarray: The state values.
        """
        values = np.zeros(self.grid.h*self.grid.w)
        for r in range(self.grid.h):
            for c in range(self.grid.w):
                state = self.grid.get_state_from_pos((r, c))
                values[state] = np.max(self.q_values[state])
        return values

# Main function
if __name__ == "__main__":
    grids = GridWorldBuilder('GridWorld.py')
    for grid in grids:
        agent = ModelFreeRL(grids)
        agent.train()
        values = agent.get_values()
        policy = agent.get_policy()
