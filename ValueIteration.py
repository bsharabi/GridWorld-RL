import numpy as np
from GridWorldIterator import *

def value_iteration(w, h, rewards, p, r, discount_factor=0.5, theta=0.01):
    
    value_function = np.zeros((h, w))
    reward_grid = np.full((h, w), r)
    
    for x, y, reward in rewards:
        reward_grid[h-1-y, x] = reward
        
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    action_names = ["up", "down", "left", "right"]
    policy = np.full((h, w), "", dtype=object)
    max_reward = max(pos[2] for pos in rewards)

    while True:
        delta = 0
        for y in range(h):
            for x in range(w):
                if (x, y) in [pos[:2] for pos in rewards]:
                    val = reward_grid[h-1-y, x]
                    if max_reward == val:
                        policy[h-y-1,x]='maxreward' 
                    else: 
                        policy[h-y-1,x]='reward' if val>0  else  'negreward' if val < 0  else 'unreward'
                        continue
                    
                v = value_function[h-1-y, x]
                values = []
                for action in actions:
                    new_y, new_x = h-1-y + action[0], x + action[1]
                    if 0 <= new_y < h and 0 <= new_x < w:
                        values.append(p * (reward_grid[new_y, new_x] + discount_factor * value_function[new_y, new_x]) + 
                                      (1 - p) / 2 * (reward_grid[h-1-y, x] + discount_factor * value_function[h-1-y, x]))
                    else:
                        values.append(reward_grid[h-1-y, x] + discount_factor * value_function[h-1-y, x])
                value_function[h-1-y, x] = max(values)
                delta = max(delta, abs(v - value_function[h-1-y, x]))
                best_action = np.argmax(values)
                policy[h-1-y, x] = action_names[best_action] if reward_grid[h-1-y, x]!=max_reward else 'maxreward' 
                
        if delta < theta:
            break

    return policy, value_function

if __name__ == "__main__":
    iterator = GridWorldIterator('GridWorld.py')
    for w ,h ,L,p,r in iterator:
        policy, value_function = value_iteration(w, h, L, p, r)
        print("Policy:\n", policy)
        print("Value Function:\n", value_function)
        
