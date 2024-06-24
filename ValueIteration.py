# import numpy as np

# def value_iteration(w, h, rewards, p, r, discount_factor=0.5, theta=0.01):
#     value_function = np.zeros((h, w))
#     reward_grid = np.full((h, w), r)
    
#     for x, y, reward in rewards:
#         reward_grid[y, x] = reward

#     actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

#     while True:
#         delta = 0
#         for y in range(h):
#             for x in range(w):
#                 if (x, y) in [(pos[0], pos[1]) for pos in rewards]:  # Skip terminal states
#                     continue
#                 v = value_function[y, x]
#                 values = []
#                 for action in actions:
#                     new_y, new_x = y + action[0], x + action[1]
#                     if 0 <= new_y < h and 0 <= new_x < w:
#                         values.append(p * (reward_grid[new_y, new_x] + discount_factor * value_function[new_y, new_x]) + 
#                                       (1 - p) / 2 * (reward_grid[y, x] + discount_factor * value_function[y, x]))
#                     else:
#                         values.append(reward_grid[y, x] + discount_factor * value_function[y, x])
#                 value_function[y, x] = max(values)
#                 delta = max(delta, abs(v - value_function[y, x]))

#         if delta < theta:
#             break

#     policy = np.zeros((h, w, 4))  # 4 possible actions: up, down, left, right
#     for y in range(h):
#         for x in range(w):
#             if (x, y) in [(pos[0], pos[1]) for pos in rewards]:
#                 continue
#             values = []
#             for action in actions:
#                 new_y, new_x = y + action[0], x + action[1]
#                 if 0 <= new_y < h and 0 <= new_x < w:
#                     values.append(p * (reward_grid[new_y, new_x] + discount_factor * value_function[new_y, new_x]) + 
#                                   (1 - p) / 2 * (reward_grid[y, x] + discount_factor * value_function[y, x]))
#                 else:
#                     values.append(reward_grid[y, x] + discount_factor * value_function[y, x])
#             best_action = np.argmax(values)
#             policy[y, x] = np.eye(4)[best_action]

#     return policy, value_function


# if __name__ == "__main__":
#     w = 4
#     h = 3
#     L = [(1, 1, 0), (3, 2, 1), (3, 1, -1)]
#     p = 0.8
#     r = -0.04
#     policy, value_function = value_iteration(w, h, L, p, r)
#     print("Policy:\n", policy)
#     print("Value Function:\n", value_function)


import numpy as np

def value_iteration(w, h, rewards, p, r, discount_factor=0.5, theta=0.01):
    value_function = np.zeros((h, w))
    reward_grid = np.full((h, w), r)
    
    for x, y, reward in rewards:
        reward_grid[h-1-y, x] = reward

    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    action_names = ["up", "down", "left", "right"]

    while True:
        delta = 0
        for y in range(h):
            for x in range(w):
                if (x, h-1-y) in [(pos[0], pos[1]) for pos in rewards]:  # Skip terminal states
                    continue
                v = value_function[y, x]
                values = []
                for action in actions:
                    new_y, new_x = y + action[0], x + action[1]
                    if 0 <= new_y < h and 0 <= new_x < w:
                        values.append(p * (reward_grid[h-1-new_y, new_x] + discount_factor * value_function[new_y, new_x]) + 
                                      (1 - p) / 2 * (reward_grid[h-1-y, x] + discount_factor * value_function[y, x]))
                    else:
                        values.append(reward_grid[h-1-y, x] + discount_factor * value_function[y, x])
                value_function[y, x] = max(values)
                delta = max(delta, abs(v - value_function[y, x]))

        if delta < theta:
            break

    policy = np.full((h, w), "", dtype=object)
    for y in range(h):
        for x in range(w):
            if (x, h-1-y) in [(pos[0], pos[1]) for pos in rewards]:
                continue
            values = []
            for action in actions:
                new_y, new_x = y + action[0], x + action[1]
                if 0 <= new_y < h and 0 <= new_x < w:
                    values.append(p * (reward_grid[h-1-new_y, new_x] + discount_factor * value_function[new_y, new_x]) + 
                                  (1 - p) / 2 * (reward_grid[h-1-y, x] + discount_factor * value_function[y, x]))
                else:
                    values.append(reward_grid[h-1-y, x] + discount_factor * value_function[y, x])
            best_action = np.argmax(values)
            policy[y, x] = action_names[best_action]

    return policy, value_function

if __name__ == "__main__":
    w = 4
    h = 3
    L = [(1, 1, 0), (3, 2, 1), (3, 1, -1)]
    p = 0.8
    r = -0.04
    policy, value_function = value_iteration(w, h, L, p, r)
    print("Policy:\n", policy)
    print("Value Function:\n", value_function)
