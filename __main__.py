from MBRL import ModelBasedRL
from MDP import ValueIteration
from MFRL import ModelFreeRL
from GridWorldBuilder import *



if __name__=="__main__":
    grids = GridWorldBuilder('GridWorld.py')
    for grid in grids:
        solver = ValueIteration(grids)
        solver.train()
        grids.visualize_value_policy(solver.policy,solver.values,True,solver.delta_history,solver.discount_factor)


        
        agentM = ModelBasedRL(grids)
        policy = agentM.iterative_policy_learning()
        values = np.max(agentM.q_values, axis=1).reshape(agentM.grid.h, agentM.grid.w)
        
        
        agentF = ModelFreeRL(grids)
        agentF.train()
        values = agentF.get_values()
        policy = agentF.get_policy()