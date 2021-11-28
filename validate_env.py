import torch
import numpy as np
import gurobipy as gp

from graph import Graph
from bssrp_mip import BSSRPMIP
from environment import Environment



def eval_mip_sol_in_env(mip, g):
    graph_dict = {0 : g}
    env = Environment(graph_dict, "test", verbose=False)
    
    env.reset(0)

    route_reward = {}
    for i, route in mip.routes.items():
        route_reward[i] = 0
        if len(route) == 0:
            continue

        for a in route[1:]:
            _, reward, done, _ = env.step(torch.tensor(a))
            route_reward[i] += reward
            if done:
                break
        if done:
            break

    total_reward = 0
    for i, reward in route_reward.items():
        total_reward += reward
    total_reward = total_reward.item() * 500
          
    return total_reward
  


def main():

    tol = 1e-3
    seed = 12343

    use_penalties = True
    no_bikes_leaving = True

    num_nodes = 10
    num_vehicles = 5
    time_limit = 30

    penalty_cost_demand = 2
    penalty_cost_time = 5
    bike_load_time = 0

    speed = 30 

    g = Graph(
            num_nodes = num_nodes, 
            k_nn = 5,
            num_vehicles = num_vehicles,
            penalty_cost_demand = penalty_cost_demand,
            penalty_cost_time = penalty_cost_time, 
            speed = speed,
            bike_load_time=bike_load_time,
            time_limit = time_limit)

    g.seed(seed)
    g.bss_graph_gen()

    ### Evaluate

    # get MIP routes/reward
    mip = BSSRPMIP(g, use_penalties=True, no_bikes_leaving=True)
    mip.optimize()

    mip_obj = mip.model.objVal
    mip_reward = - eval_mip_sol_in_env(mip, g)

    if (np.abs(mip_reward - mip_obj) > tol):
        raise Exception(f"MIP objective value and reward differ.  Obj {mip_obj}, Reward {mip_reward}")

    print("MIP objective value and reward are the same.")


if __name__ == "__main__":
    main()











