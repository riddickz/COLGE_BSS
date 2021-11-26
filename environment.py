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
        self.static = self.graph.static.detach()
        self.state = self.compute_state()
        self.prev_node = 0
        self.t_total = 0.
        self.tour_indices = [0]
        self.prev_demand = np.abs(self.dynamic[2, 1:]).sum()
        self.trip_count = 0
        self.edge_index, _ = self.graph.get_edge()
        return self.state, self.graph.W

    def compute_state(self):
        state = torch.cat((self.dynamic,self.static.T),dim=0)
        return state.float()

    def demand_reset(self):
        self.graph.refresh_demand()
        self.dynamic = self.graph.dynamic.detach().clone()
        self.dynamic_init = self.dynamic.detach().clone()
        self.state = self.compute_state()
        return self.state

    def step(self, action):
        done = False
        chosen_idx = action.item()

        if chosen_idx == 0:
            new_load = 0
            new_demand = 0
        else:
            new_load, new_demand = self.get_updated_load_and_demand(chosen_idx)

        reward = self.get_reward(chosen_idx)

        # Updates the dynamic(observation, load, demand)
        self.dynamic[0, chosen_idx] = 1
        self.dynamic[1] = new_load
        self.dynamic[2, chosen_idx] = new_demand
        self.dynamic[3, :] = 0  # current node
        self.dynamic[3, chosen_idx] = 1
        self.dynamic[4, :] = 0  # previous node
        self.dynamic[4, self.prev_node] = 1

        # returning to depot
        #elif (self.dynamic[5].sum() >= self.graph.time_limit) or chosen_idx == 0:
        if chosen_idx == 0:
            self.dynamic[5, :] = 0
            back_depot = True

        # continuing on route
        else:
            self.dynamic[5, chosen_idx] = self.get_travel_dist(self.prev_node, chosen_idx)
            back_depot = False

        self.t_total += self.get_travel_dist(self.prev_node, chosen_idx)
        self.tour_indices.append(chosen_idx)
        self.prev_node = chosen_idx

        if chosen_idx == 0:
            self.trip_count += 1

        info = (self.prev_node, self.t_total, self.tour_indices, back_depot)

        # terminal case
        if (self.dynamic[0] != 0).all():

            reward += self.get_terminal_reward(chosen_idx, new_load)

            self.t_total += self.get_travel_dist(chosen_idx, 0)
            self.tour_indices.append(0)
            done = True

            print("Tour: ", self.tour_indices)
            print("Tour Time: ", self.t_total.item())
            print("Left Demand: ", new_demand)
            print("Node Visits: ", len(self.tour_indices))
            print("Games Finished: ", self.games)
            print("#" * 100)

        self.state = self.compute_state()
        return (self.state, reward, done, info)


    def get_terminal_reward(self, chosen_idx, excess):
        """ Gets the reward when terminal state is reached. """
        reward = 0
        reward += self.get_travel_dist(chosen_idx, 0) # time to go back to depot
        reward += excess * self.graph.penalty_cost_demand # additional bikes on vehicle
        reward += self.get_overage_last_step(chosen_idx)  * self.graph.penalty_cost_time # overtime
        return - reward

    def get_reward(self, chosen_idx):
        """ Gets the reward action.  """
        reward = 0
        reward += self.get_travel_dist(self.prev_node, chosen_idx) # travel time from prev node to next node
        reward += self.get_demand_reward(chosen_idx) * self.graph.penalty_cost_demand # difference in unmet demand
        reward += self.get_overage_time(chosen_idx) * self.graph.penalty_cost_time # overtime
        return - reward

    def get_overage_last_step(self, chosen_idx):
        """ Gets the overage time for moving to the depot in the last step.  """
        dist_to_idx = self.get_current_route_time() + self.get_travel_dist(self.prev_node, chosen_idx)
        dist_to_depot = self.get_current_route_time() + self.get_travel_dist(self.prev_node, chosen_idx)
        dist_to_depot += self.get_travel_dist(chosen_idx, 0)

        if dist_to_idx > self.graph.time_limit:
            return self.get_travel_dist(chosen_idx, 0)
        elif dist_to_depot > self.graph.time_limit:
            return dist_to_depot - self.graph.time_limit
        else:
            return 0

    def get_overage_time(self, chosen_idx):
        """ Gets the overage time for moving a node.  """
        if self.get_current_route_time() > self.graph.time_limit:
            return self.get_travel_dist(self.prev_node, chosen_idx)
        elif self.get_current_route_time() + self.get_travel_dist(self.prev_node, chosen_idx) > self.graph.time_limit:
            return self.get_current_route_time() + self.get_travel_dist(self.prev_node, chosen_idx) - self.graph.time_limit
        else:
            return 0

    def get_travel_dist(self, cur_node, next_node):
        """ Gets the travel distance between two nodes.  """
        return self.graph.W_full[cur_node, next_node]

    def get_current_route_time(self):
        """ Gets the current route time. """
        return self.dynamic[5].sum().item()

    def get_updated_load_and_demand(self, chosen_idx):
        """ Gets the updated load and demands. """
        return self._get_new_load_demand(chosen_idx)

    def get_demand_reward(self, chosen_idx):
        """ Gets the unmet demand at a current node or load if returning to depot. """
        load, demand = self._get_new_load_demand(chosen_idx)
        if chosen_idx == 0:
            return load
        else:
            return np.abs(demand)

    def _get_new_load_demand(self, chosen_idx):
        """ Gets the new load and demand from visiting chosen_idx. """
        # difference in unmet demand
        load_idx = self.dynamic[1].clone()[chosen_idx]
        demand_idx = self.graph.demands[chosen_idx]

        new_load = torch.clamp(load_idx + demand_idx, max=self.graph.max_load, min=0)
        load_diff = new_load - load_idx
        new_demand = demand_idx - load_diff

        return new_load, new_demand


    #     chosen_idx = action.item()
    #     t, reward, done = self.get_reward(chosen_idx)
    #
    #     # Clone the dynamic variable so we don't mess up graph
    #     all_loads = self.dynamic[1].clone()
    #     all_demands = self.dynamic[2].clone()
    #
    #     load = all_loads[chosen_idx]
    #     demand = all_demands[chosen_idx]
    #
    #     # Loading and supplying constraint
    #     # if we've chosen to visit a city, try to satisfy as much demand as possible
    #     new_load = torch.clamp(load - demand, max=self.graph.max_load, min=0)
    #     if demand >= 0:
    #         new_demand = torch.clamp(demand - load, min=0)
    #     else:
    #         new_demand = torch.clamp(demand + (self.graph.max_load - load), max=0)
    #
    #     # # Updates the dynamic(observation, load, demand)m node_idx,tour values
    #     # if all_demands[chosen_idx] == 0:
    #     #     self.dynamic[0, chosen_idx] = 1  # node is covered if demand is met
    #     # else:
    #     #     self.dynamic[0, chosen_idx] = 0
    #
    #     self.dynamic[0, chosen_idx] = 1
    #     self.dynamic[1] = new_load
    #     self.dynamic[2, chosen_idx] = new_demand
    #     self.dynamic[3, :] = 0  # current node
    #     self.dynamic[3, chosen_idx] = 1
    #     self.dynamic[4, :] = 0  # previous node
    #     self.dynamic[4, self.prev_node] = 1
    #
    #     if (self.dynamic[5].sum() >= self.graph.time_limit):
    #         self.dynamic[5, :] = 0
    #         back_depot = True
    #     else:
    #         self.dynamic[5, chosen_idx] = t  # previous node
    #         back_depot = False
    #
    #     self.prev_node = chosen_idx
    #     self.t_total += t.item()
    #     self.tour_indices.append(chosen_idx)
    #     if chosen_idx == 0:
    #         self.trip_count += 1
    #
    #     info = (self.prev_node, self.t_total, self.tour_indices, back_depot)
    #
    #     self.state = self.compute_state()
    #
    #     return (self.state, reward, done, info)
    #
    # def get_reward(self, chosen_idx):
    #     done = False
    #     prev_node = torch.where(self.dynamic[4] == 1)
    #
    #     t = self.graph.W_weighted[chosen_idx, prev_node]
    #
    #     demand = np.abs(self.dynamic[2, 1:]).sum()
    #     # demand_chg = self.prev_demand - demand
    #     # reward = - t + demand_chg
    #     reward = - t
    #     # print(t, demand_chg)
    #
    #     self.prev_demand = demand
    #
    #     # Terminal State
    #     timeout = bool(self.t_total >= self.graph.time_limit * self.graph.num_vehicles)
    #     full_visit = (self.dynamic[0] != 0).all()
    #
    #     if timeout or full_visit or demand == 0: #or self.trip_count ==3:
    #         time_depot = self.graph.W_weighted[0, chosen_idx]  # travel back to depot
    #         reward -= (time_depot + 3*demand)  # t_ij * x_ijk + β(p_b+ + p_b−)
    #         self.t_total += time_depot
    #         self.tour_indices.append(0)
    #         done = True
    #
    #         if demand == 0:
    #             print("Zero Demands!")
    #         elif timeout:
    #             print("Time Out!")
    #         elif full_visit:
    #             print("Full Visit!")
    #
    #         print("Tour: ", self.tour_indices)
    #         print("Tour Time Cost: ", self.t_total.item())
    #         print("Left Demand: ", demand)
    #         print("Node Visits: ", len(self.tour_indices))
    #         print("Games Finished: ", self.games)
    #         print("#" * 100)
    #     # print("visit node:{}, reward:{}, done:{}:".format(chosen_idx, reward, done))
    #     # print("#"*100)
    #     return t, reward, done

    def render(self, save_path=None):
        """Plots the found solution."""
        plt.ion()
        plt.figure(0, figsize=(7, 7))

        nodes = self.graph.static.numpy()
        W = self.graph.W.detach().numpy()

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
