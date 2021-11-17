import numpy as np
import torch
import matplotlib.pyplot as plt

"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self, graph_dict, name):
        self.graph_dict = graph_dict
        self.name = name

    def reset(self, g):
        self.games = g
        self.graph = self.graph_dict[self.games]
        self.dynamic = self.graph.dynamic.detach().clone()
        self.dynamic_init = self.dynamic.detach().clone()
        self.prev_node = 0
        self.t_total = 0.
        self.tour_indices = [0]
        self.prev_demand = np.abs(self.dynamic[2, 1:]).sum()

        return self.dynamic, self.graph.W_weighted  # aka state

    def step(self, action):
        chosen_idx = action.item()
        t, reward, done = self.get_reward(chosen_idx)

        # Clone the dynamic variable so we don't mess up graph
        all_loads = self.dynamic[1].clone()
        all_demands = self.dynamic[2].clone()

        load = all_loads[chosen_idx]
        demand = all_demands[chosen_idx]

        # Loading and supplying constraint
        # if we've chosen to visit a city, try to satisfy as much demand as possible
        new_load = torch.clamp(load - demand, max=self.graph.max_load, min=0)
        if demand >= 0:
            new_demand = torch.clamp(demand - load, min=0)
        else:
            new_demand = torch.clamp(demand + (self.graph.max_load - load), max=0)

        # Updates the dynamic(observation, load, demand)m node_idx,tour values
        if all_demands[chosen_idx] == 0:
            self.dynamic[0, chosen_idx] = 1  # node is covered if demand is met
        else:
            self.dynamic[0, chosen_idx] = 0

        self.dynamic[1] = new_load
        self.dynamic[2, chosen_idx] = new_demand
        self.dynamic[3, :] = 0  # current node
        self.dynamic[3, chosen_idx] = 1
        self.dynamic[4, :] = 0  # previous node
        self.dynamic[4, self.prev_node] = 1

        if (self.dynamic[5].sum() >= self.graph.time_limit):
            self.dynamic[5, :] = 0
            back_depot = True
        else:
            self.dynamic[5, chosen_idx] = t  # previous node
            back_depot = False

        self.prev_node = chosen_idx
        self.t_total += t.item()
        self.tour_indices.append(chosen_idx)

        info = (self.prev_node, self.t_total, self.tour_indices, back_depot)

        return (self.dynamic, reward, done, info)

    def get_reward(self, chosen_idx):
        done = False
        prev_node = torch.where(self.dynamic[4] == 1)

        t = self.graph.W_weighted[chosen_idx, prev_node]

        demand = np.abs(self.dynamic[2, 1:]).sum()
        demand_chg = self.prev_demand - demand
        reward = - t + demand_chg
        # print(t, demand_chg)

        self.prev_demand = demand

        # Terminal State
        timeout = bool(self.t_total >= self.graph.time_limit * self.graph.num_vehicles)
        full_visit = (self.dynamic[0] != 0).all()

        if timeout or full_visit or demand == 0:
            time_depot = self.graph.W_weighted[0, chosen_idx]  # travel back to depot
            reward -= (time_depot + demand)  # t_ij * x_ijk + β(p_b+ + p_b−)
            self.t_total += time_depot
            self.tour_indices.append(0)
            done = True

            if timeout:
                print("TIMEOUT!")

            print("Tour: ", self.tour_indices)
            print("Tour Time Cost: ", self.t_total.item())
            print("Left Demand: ", demand)
            print("Node Visits: ", len(self.tour_indices))
            print("Games Finished: ", self.games)
            print("#" * 100)
        # print("visit node:{}, reward:{}, done:{}:".format(chosen_idx, reward, done))
        # print("#"*100)
        return t, reward, done

    def render(self, save_path=None):
        """Plots the found solution."""
        plt.ion()
        plt.figure(0, figsize=(7, 7))

        nodes = self.graph.static.numpy()
        W = self.graph.W

        # Plot nodes
        colors = ['red']  # First node as depot
        for i in range(len(nodes) - 1):
            colors.append('blue')

        xs, ys = nodes[:, 0], nodes[:, 1]
        plt.scatter(xs, ys, color=colors)

        # Plot edges
        edgeSet = set()
        for row in range(W.shape[0]):
            for column in range(W.shape[1]):
                if W.item(row, column) == 1 and (column, row) not in edgeSet:  # get rid of repeat edge
                    edgeSet.add((row, column))

        for edge in edgeSet:
            X = nodes[edge, 0]
            Y = nodes[edge, 1]
            plt.plot(X, Y, "g-", lw=2, alpha=0.1)

        # Plot tours
        for i, idx in enumerate(self.tour_indices):
            if i < len(self.tour_indices) - 1:
                next_node = self.tour_indices[i + 1]
                X = [nodes[idx][0], nodes[next_node][0]]
                Y = [nodes[idx][1], nodes[next_node][1]]
            else:
                X = [nodes[idx][0], nodes[0][0]]
                Y = [nodes[idx][1], nodes[0][1]]

            plt.plot(X, Y, "black", lw=0.3)

        # Show dynamic
        for i, (x, y) in enumerate(zip(xs, ys)):
            label = "N{}: {}/{}".format(i, self.dynamic[2][i].int(), self.dynamic_init[2][i].int())
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 5),  # distance from text to points (x,y)
                         ha='center',
                         fontsize=5,
                         color="orange")

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("Game {} Time Cost: {}".format(self.games,round(self.t_total.item())))
        plt.pause(0.001)
        plt.close()

        # plt.savefig(save_path, bbox_inches='tight', dpi=200)
