import numpy as np
from GridWorldBuilder import *
import matplotlib.pyplot as plt

class ValueIteration:
    """
    Class for performing value iteration on a Markov Decision Process.

    Attributes:
        reward_function (np.ndarray): The reward function for each state.
        transition_model (np.ndarray): The transition model of the MDP.
        discount_factor (float): The discount factor for future rewards.
        theta (float): The threshold for stopping the iteration.
    """

    def __init__(self,grid:GridWorldBuilder, discount_factor=0.5, theta=0.01):
        """
        Initializes the ValueIteration class with the given parameters.

        Args:
            reward_function (np.ndarray): The reward function for each state.
            transition_model (np.ndarray): The transition model of the MDP.
            discount_factor (float, optional): The discount factor for future rewards. Default is 1.
            theta (float, optional): The threshold for stopping the iteration. Default is 0.01.
        """
        self.num_states = grid.num_states
        self.num_actions = grid.num_actions
        self.reward_function = grid.reward_table
        self.transition_model = grid.transition
        self.discount_factor = discount_factor
        self.values = np.zeros(self.num_states)
        self.policy = None
        self.grid =grid
        self.theta = theta
    
    def one_iteration(self,one=True):
        """
        Performs one iteration of value iteration.

        Returns:
            float: The maximum change in value during this iteration.
        """
        delta = 0

        for s in range(self.num_states):
            v = self.values[s]
            if s in self.grid.L and not one:
                continue
            self.values[s] = max(sum(self.transition_model[s, a, s_next] * 
                           (self.reward_function[s] + self.discount_factor * self.values[s_next])
                           for s_next in range(self.num_states)) for a in range(self.num_actions))
            delta = max(delta, abs(v - self.values[s]))
        return delta 
    
    def get_policy(self):
        """
        Extracts the policy from the value function.

        Returns:
            np.ndarray: The policy for each state.
        """
        policy = np.zeros(self.num_states)
        for s in range(self.num_states):
            policy[s]= max(range(self.num_actions), 
                            key=lambda a: sum(
                            self.transition_model[s, a, s_next] *
                                                    (self.reward_function[s] + self.discount_factor * self.values[s_next])
                                                    for s_next in range(self.num_states)))
            
        return policy.astype(int)
   

    def train(self):
        """
        Trains the value iteration model until convergence.

        Args:
            plot (bool, optional): Whether to plot the delta values over iterations. Default is False.
        """

        epoch = 0
        delta = self.one_iteration()
        delta_history = [delta]
        while delta > self.theta:
            epoch += 1
            delta = self.one_iteration(False)
            delta_history.append(delta)
            if delta < self.theta:
                break
        self.policy = self.get_policy()
        self.delta_history=delta_history
        

if __name__ == "__main__":
    grids = GridWorldBuilder('GridWorld.py')
    for grid in grids:
        solver = ValueIteration(grids)
        solver.train()
        grids.visualize_value_policy(solver.policy,solver.values,True,solver.delta_history,solver.discount_factor)

