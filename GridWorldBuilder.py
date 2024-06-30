import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
class GridWorldBuilder:
    """
    Class for building and managing grid worlds for reinforcement learning tasks.

    Attributes:
        filename (str): The name of the file containing grid world definitions.
        grids (list): A list of parsed grid worlds.
        current_grid (dict): The current grid world being used.
        current_index (int): Index of the current grid world.
    """

    def __init__(self, filename):
        """
        Initializes the GridWorldBuilder with the given filename.
        
        Args:
            filename (str): The name of the file containing grid world definitions.
        """
        self.filename = filename
        self.grids = self._parse_file()
        self.current_grid = None
        self.current_index = -1

    def update_attribute(self):
        """
        Updates attributes based on the current grid world.
        """
        self.h = self.current_grid['h']
        self.w = self.current_grid['w']
        self.L = self.current_grid['L']
        self.r = self.current_grid['r']
        self.p = self.current_grid['p']
        self.num_states = self.h * self.w
        self.num_actions = 4

    def _parse_file(self):
        """
        Parses the file to extract grid world definitions.
        
        Returns:
            list: A list of dictionaries representing grid worlds.
        """
        grids = []
        with open(self.filename, 'r') as file:
            content = file.read()
            grid_blocks = content.split('\n\n')  
            for block in grid_blocks:
                if block.strip():
                    grid = {}
                    for line in block.split('\n'):
                        line = line.strip()
                        if line.startswith('#') or not line:
                            continue
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            value = ast.literal_eval(value)
                        except ValueError:
                            pass
                        grid[key] = value
                    grid['reward']=grid['L']
                    grid['L'] = {pos[0] + (grid['h']-pos[1]-1) * grid['w'] : pos[2] for pos in grid['L']}
                    
                    grids.append(grid)
        return grids

    def transition_model(self):
        """
        Generates the transition model for the current grid world.
        """
        transition_model = np.zeros((self.num_states, self.num_actions, self.num_states))
        left_right = round((1 - self.p) * 10) / 10 / 2
        for r in range(self.h):
            for c in range(self.w):
                s = self.get_state_from_pos((r, c))
                neighbor_s = np.zeros(self.num_actions)
                
                if self.map[s] == self.r and s not in self.L.items():
                    for a in range(self.num_actions):
                        new_r, new_c = r, c
                        if a == 0:
                            new_r = max(r - 1, 0)
                        elif a == 1:
                            new_c = min(c + 1, self.w - 1)
                        elif a == 2:
                            new_r = min(r + 1, self.h - 1)
                        elif a == 3:
                            new_c = max(c - 1, 0)
                        if self.map[self.get_state_from_pos((new_r, new_c))] == 0:
                            new_r, new_c = r, c
                        s_prime = self.get_state_from_pos((new_r, new_c))
                        neighbor_s[a] = s_prime
                else:
                    neighbor_s = np.ones(self.num_actions) * s
                for a in range(self.num_actions):
                    transition_model[s, a, int(neighbor_s[a])] += self.p
                    transition_model[s, a, int(neighbor_s[(a + 1) % self.num_actions])] += left_right
                    transition_model[s, a, int(neighbor_s[(a - 1) % self.num_actions])] += left_right
        self.transition = transition_model

    def get_state_from_pos(self, pos):
        """
        Converts a position to a state index.
        
        Args:
            pos (tuple): A tuple representing the position (row, column).
        
        Returns:
            int: The state index.
        """
        return pos[0] * self.w + pos[1]

    def get_pos_from_state(self, state):
        """
        Converts a state index to a position.
        
        Args:
            state (int): The state index.
        
        Returns:
            tuple: A tuple representing the position (row, column).
        """
        return state // self.w, state % self.w   
    
    def reward_function(self):
        """
        Generates the reward function for the current grid world.
        """
        self.reward_table = np.zeros(self.num_states)
        for r in range(self.h):
            for c in range(self.w):
                s = self.get_state_from_pos((r, c))
                self.reward_table[s] = self.map[s]
          
    def __iter__(self):
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index >= len(self.grids):
            raise StopIteration
        self.current_grid = self.grids[self.current_index]
        self.update_attribute()
        self.map = np.full(self.h * self.w, self.r)
        for state, reward in self.L.items():
            self.map[state] = reward
        self.transition_model()
        self.reward_function()
        return self.w, self.h, self.L, self.p, self.r

    def __prev__(self):
        self.update_attribute()
        pass
    

    def visualize_value_policy(self, policy, values, plot,delta_history,discount_factor,fig_size=(8, 6)):
        unit = min(fig_size[1] // self.h, fig_size[0] // self.w)
        unit = max(1, unit)
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.axis('off')

        for i in range(self.w + 1):
            if i == 0 or i == self.w:
                ax.plot([i * unit, i * unit], [0, self.h * unit],
                        color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.h * unit],
                        alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.h + 1):
            if i == 0 or i == self.h:
                ax.plot([0, self.w * unit], [i * unit, i * unit],
                        color='black')
            else:
                ax.plot([0, self.w * unit], [i * unit, i * unit],
                        alpha=0.7, color='grey', linestyle='dashed')

        for i in range(self.h):
            for j in range(self.w):
                y = (self.h - 1 - i) * unit
                x = j * unit
                s = self.get_state_from_pos((i, j))
                if self.map[s] == 0:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif s in self.L and self.L[s] == self.map[s] and self.map[s] <0:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif s in self.L and self.L[s] == self.map[s] and self.map[s] >0 :
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green',
                                             alpha=0.6)
                    ax.add_patch(rect)
                if self.map[s] != 0:
                    ax.text(x + 0.5 * unit, y + 0.5 * unit, f'{values[s]:.4f}',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=max(fig_size)*unit*0.6)
                if policy is not None:
                    if self.map[s] != 0 and s not in self.L :
                        a = policy[s]
                        symbol = ['^', '>', 'v', '<']
                        ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=symbol[a], alpha=0.4,
                                linestyle='none', markersize=max(fig_size)*unit, color='#1f77b4')

        plt.tight_layout()
        plt.show()


        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
            ax.plot(np.arange(len(delta_history)) + 1, delta_history, marker='o', markersize=4,
                    alpha=0.7, color='#2ca02c', label=r'$\gamma= $' + f'{discount_factor}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Delta')
            ax.legend()
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    g = GridWorldBuilder("GridWorld.py")
    for _ in g:
        for current_s in range(g.num_states):
            for next_s in range(g.num_states):
                for action in range(g.num_actions):
                    print(f'P(s\' = {next_s}| s = {current_s} ,a = {action} ) = {g.transition[current_s, action,next_s ]}')
        print(g.reward_table)
        input()
