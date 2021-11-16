import numpy as np
import torch
import pulp
import networkx as nx
import matplotlib.pyplot as plt
"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self, graph,name):
        self.graphs = graph
        self.name= name

    def reset(self, g):
        self.games = g
        self.graph_init = self.graphs[self.games]
        self.nodes = self.graph_init.nodes()
        self.nbr_of_nodes = 0
        self.edge_add_old = 0
        self.last_reward = 0
        self.observation = torch.zeros(1,self.nodes,3,dtype=torch.float)
        self.static = self.graph_init.static # location coordinates
        self.dynamic = self.graph_init.dynamic.int().clone()# load, demand
        self.load_init = self.dynamic[0].clone()
        self.demand_init = self.dynamic[1].clone()

        self.prev_demands = self.graph_init.dynamic[1].clone()
        self.observation[:,:,2] = self.dynamic[0] # load
        self.observation[:,:,2] = self.dynamic[1] #
        self.demand_init_tot = torch.abs(self.dynamic[1][1:]).sum()
        self.max_load = self.graphs[self.games].max_load
        self.prev_node = 0
        self.tour_indices = []
        self.tour_length = 0.

    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        return self.observation

    def step(self, chosen_idx):

        # # Update the dynamic elements differently for if we visit depot vs. a city
        # chosen_idx = torch.tensor(chosen_idx)
        # visit = chosen_idx.ne(0)
        # depot = chosen_idx.eq(0)

        # Clone the dynamic variable so we don't mess up graph
        all_loads = self.dynamic[0].clone()
        all_demands = self.dynamic[1].clone()

        load = all_loads[chosen_idx]
        demand = all_demands[chosen_idx]

        # Loading and supplying constraint
        # if we've chosen to visit a city, try to satisfy as much demand as possible
        new_load = torch.clamp(load - demand, max= self.max_load, min=0)
        if demand >= 0:
            new_demand = torch.clamp(demand - load, min=0)
        else:
            new_demand = torch.clamp(demand + (self.max_load - load), max = 0)

        all_loads[:] = new_load #TODO
        all_demands[chosen_idx] = new_demand

        # print("all_demands:" ,all_demands)
        # print("all_loads:" ,all_loads)

        # Updates the (observation, load, demand,node_idx,tour) values
        if all_demands[chosen_idx] == 0:
            self.observation[:, chosen_idx, 0] = 1 # node is covered if demand is met
        else:
            self.observation[:, chosen_idx, 0] = new_demand/self.graph_init.dynamic[1][chosen_idx]

        self.observation[:, :, 1] = new_load
        self.observation[:, chosen_idx, 2] = new_demand

        self.dynamic[0] = all_loads
        self.dynamic[1] = all_demands

        reward = self.get_reward(chosen_idx)

        self.prev_demands[chosen_idx] = new_demand
        self.prev_node = chosen_idx
        self.tour_indices.append(chosen_idx)
        return reward

    def get_reward(self, chosen_idx):
        done=False
        loc_prev = self.static[self.prev_node]
        loc = self.static[chosen_idx]
        dist = (loc_prev-loc).norm()
        self.tour_length += dist

        demand = np.abs(self.dynamic[1][1:]).sum()
        demand_change = np.abs(self.dynamic[1] - self.prev_demands).sum()

        reward = - dist/30*120 + demand_change


        dist_limit = bool(self.tour_length >= 3*60) #TODO: total dist limit: 3 car x 120 min at 30km/h

        # Termination Criteria
        if demand == 0 or (self.observation[:, :, 0] != 0).all() or dist_limit:
            if dist_limit:
                print("TIMEOUT")

            dist_depot= (loc - self.static[0]).norm() # add distance back to depot
            self.tour_length += dist_depot

            demand_penalty = torch.abs(self.demand_init - self.dynamic[1]).sum()

            reward -= (dist_depot/30*120 + demand_penalty) # t_ij * x_ijk + β(p_b+ + p_b−)
            done = True

            print("Tour: ", self.tour_indices)
            print("Tour Length: ", self.tour_length.item() )
            print("Node Visits: ", len(self.tour_indices))
            print("Games Finished: ", self.games )
            print("#"*100)
            # visualize_2D(self.graph_init.static.numpy(), self.graph_init.W)
            # nx.draw(self.graph_init.g, with_labels=True)
            # plt.show()
        # print("demand_penalty: ", demand_penalty)
        # print("visit node:{}, reward:{}, done:{}:".format(chosen_idx, reward, done))
        # print("#"*100)
        return (reward, done)

    def render(self, save_path=None):
        """Plots the found solution."""
        plt.ion()
        plt.figure(0, figsize=(7, 7))

        nodes = self.graph_init.static.numpy()
        W = self.graph_init.W

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
            label = "N{}: {}/{}".format(i, self.dynamic[1][i].int(),self.graph_init.dynamic[1][i].int())
            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 5),  # distance from text to points (x,y)
                         ha='center',
                         fontsize=5,
                         color="orange")

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.pause(0.001)
        plt.close()

        # plt.savefig(save_path, bbox_inches='tight', dpi=200)


