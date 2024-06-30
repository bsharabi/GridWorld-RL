import numpy as np
from GridWorldBuilder import GridWorldBuilder
import random

random.seed(42)

class ModelBasedRL:
    """
    Model-Based Reinforcement Learning (MBRL) agent for solving grid world problems.
    """
    
    def __init__(self, grid: GridWorldBuilder, discount_factor=0.5, epsilon=0.01, decay=0.99, learning_rate=0.01, episodes=1000) -> None:
        """
        Initialize the MBRL agent with given parameters.

        Args:
            grid (GridWorldBuilder): The grid world builder instance.
            discount_factor (float): The discount factor for future rewards.
            epsilon (float): The probability for exploration in the Boltzmann exploration policy.
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
        self.rewards = self._initialize_rewards()
        self.actions = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        self.q_values = np.zeros((self.grid.num_states, self.grid.num_actions))
        self.reward_grid = self._create_reward_grid()

    def _initialize_rewards(self):
        """
        Initialize the reward function for the grid world.

        Returns:
            list: A list of rewards for each state in the grid world.
        """
        rewards = [(x, self.grid.h-1-y, val) for x, y, val in self.grid.current_grid['reward']]
        return rewards

    def _create_reward_grid(self):
        """
        Create a grid representing the rewards for each state.

        Returns:
            np.ndarray: The reward grid.
        """
        reward_grid = np.full((self.grid.h, self.grid.w), -1)
        for x, y, val in self.rewards:
            reward_grid[y][x] = val
        return reward_grid

    def boltzmann_exploration(self, state, temperature):
        """
        Boltzmann exploration policy to select an action.

        Args:
            state (int): The current state index.
            temperature (float): The temperature parameter for Boltzmann exploration.

        Returns:
            int: The selected action.
        """
        q_values = self.q_values[state]
        exp_q = np.exp(np.array(q_values) / temperature)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(range(self.grid.num_actions), p=probs)

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

    def calculate_expected_utility(self, state, action):
        """
        Calculate the expected utility of a given state and action.

        Args:
            state (tuple): The current state coordinates.
            action (int): The action taken.

        Returns:
            float: The expected utility.
        """
        expected_utility = -1
        for delta_action in [-1, 0, 1]:
            new_action = (action + delta_action) % self.grid.num_actions
            prob = self.grid.p if delta_action == 0 else (1 - self.grid.p) / 2
            next_state = self.get_next_state(state, new_action)
            expected_utility += prob * (self.reward_grid[self.grid.get_pos_from_state(next_state)] + self.discount_factor * np.max(self.q_values[next_state]))
        return expected_utility

    def learn_mdp_from_experience(self, experience):
        """
        Learn the MDP model from experience.

        Args:
            experience (list): The list of experiences.

        Returns:
            tuple: The transition and reward matrices.
        """
        T = np.zeros((self.grid.h, self.grid.w, self.grid.num_actions, self.grid.h, self.grid.w))
        R = np.zeros((self.grid.h, self.grid.w, self.grid.num_actions))
        N = np.zeros((self.grid.h, self.grid.w, self.grid.num_actions))
        
        for (i, a, r, j) in experience:
            i_r, i_c = self.grid.get_pos_from_state(i)
            j_r, j_c = self.grid.get_pos_from_state(j)
            T[i_r, i_c, a, j_r, j_c] += 1
            R[i_r, i_c, a] += r
            N[i_r, i_c, a] += 1

        for r in range(self.grid.h):
            for c in range(self.grid.w):
                for a in range(self.grid.num_actions):
                    if N[r, c, a] > 0:
                        R[r, c, a] /= N[r, c, a]
                        T[r, c, a] /= N[r, c, a]
                        
        return T, R

    def value_iteration(self, T, R, threshold=0.01):
        """
        Perform value iteration to solve the MDP.

        Args:
            T (np.ndarray): The transition matrix.
            R (np.ndarray): The reward matrix.
            threshold (float): The threshold for convergence.

        Returns:
            tuple: The optimal policy and value function.
        """
        V = np.zeros((self.grid.h, self.grid.w))
        policy = np.zeros((self.grid.h, self.grid.w), dtype=int)
        while True:
            delta = 0
            for r in range(self.grid.h):
                for c in range(self.grid.w):
                    v = V[r, c]
                    V[r, c] = max(sum(T[r, c, a, r2, c2] * (R[r, c, a] + self.discount_factor * V[r2, c2])
                                    for r2 in range(self.grid.h) for c2 in range(self.grid.w)) for a in range(self.grid.num_actions))
                    policy[r, c] = np.argmax([sum(T[r, c, a, r2, c2] * (R[r, c, a] + self.discount_factor * V[r2, c2])
                                                for r2 in range(self.grid.h) for c2 in range(self.grid.w)) for a in range(self.grid.num_actions)])
                    delta = max(delta, abs(v - V[r, c]))
            if delta < threshold:
                break
        return policy, V

    def iterative_policy_learning(self):
        """
        Perform iterative policy learning with Boltzmann exploration.

        Returns:
            np.ndarray: The optimal policy.
        """
        experience = []
        temperature = 1
        k = 0

        while True:
            k += 1
            state = random.randint(0, self.grid.num_states - 1)
            while self.grid.get_pos_from_state(state) in [(py, px) for px, py, val in self.rewards if val != 0]:
                state = random.randint(0, self.grid.num_states - 1)

            for _ in range(10000):  # Choose a suitable number of steps for each episode
                action = self.boltzmann_exploration(state, temperature)
                next_state = self.get_next_state(state, action)
                reward = self.reward_grid[self.grid.get_pos_from_state(next_state)]
                experience.append((state, action, reward, next_state))
                state = next_state

            T, R_mdp = self.learn_mdp_from_experience(experience)
            new_policy, _ = self.value_iteration(T, R_mdp, self.discount_factor)
            policy_stable = True

            # Update Q-values and check if the policy is stable
            for r in range(self.grid.h):
                for c in range(self.grid.w):
                    for a in range(self.grid.num_actions):
                        q_value = self.calculate_expected_utility(self.grid.get_state_from_pos((r, c)), a)
                        if abs(self.q_values[self.grid.get_state_from_pos((r, c))][a] - q_value) > 0.01:
                            policy_stable = False
                        self.q_values[self.grid.get_state_from_pos((r, c))][a] = q_value

            if policy_stable:
                break
            temperature *= self.decay

        return new_policy


# Main function
if __name__ == "__main__":
    grids = GridWorldBuilder('GridWorld.py')
    for grid in grids:
        agent = ModelBasedRL(grids)
        policy = agent.iterative_policy_learning()
        values = np.max(agent.q_values, axis=1).reshape(agent.grid.h, agent.grid.w)

