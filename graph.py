import numpy as np
import networkx as nx
import collections
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os
import torch
from scipy import sparse
from torch_geometric.utils import from_scipy_sparse_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# seed = np.random.seed(120)

class Graph:
    def __init__(self,
                 num_nodes,
                 k_nn,
                 num_vehicles,
                 penalty_cost_demand,
                 penalty_cost_time,
                 speed,
                 time_limit,
                 max_load=20,
                 max_demand=9,
                 area=10):

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        self.num_nodes = num_nodes
        self.num_neighbors = k_nn
        self.max_load = max_load
        self.max_demand = max_demand
        self.area = area  # km
        self.num_vehicles = num_vehicles
        self.penalty_cost_demand = penalty_cost_demand
        self.penalty_cost_time = penalty_cost_time
        self.speed = speed
        self.time_limit = time_limit

        self.rng = np.random.RandomState()
        self.seed_used = None
        self.bss_graph_gen()

    def seed(self, _seed):
        self.seed_used = _seed
        self.rng.seed(_seed)

    def gen_instance(self):  # Generate random instance

        self.locations = self.rng.rand(self.num_nodes, 2) * self.area  # node num with (dimension) coordinates in [0,1]
        # pca = PCA(n_components=2)  # center & rotate coordinates
        # locations = pca.fit_transform(coords)
        self.refresh_demand()

    def refresh_demand(self):
        self.demands = self.get_demands()
        demands_tensor = torch.tensor(self.demands)
        cur_node = torch.zeros(self.num_nodes)
        prev_node = torch.zeros(self.num_nodes)
        cur_node[0] = 1
        prev_node[0] = 1
        trip_time = torch.zeros(self.num_nodes)
        loads = torch.zeros(self.num_nodes)

        self.static = torch.tensor(self.locations)
        self.observation = torch.zeros(self.num_nodes)
        self.dynamic = torch.stack((self.observation, loads, demands_tensor, cur_node, prev_node,trip_time),
                                   dim=0)


    def adjacenct_gen(self, num_nodes, num_neighbors, coords):
        assert num_neighbors < num_nodes

        # add KNN edges with random K
        W_val = squareform(pdist(coords, metric='euclidean'))
        W_val = self.get_time_based_distance_matrix(W_val)
        self.W_full = W_val.copy()

        W = np.zeros((num_nodes, num_nodes))
        knns = np.argpartition(W_val, kth=num_neighbors, axis=-1)[:, num_neighbors::-1]

        # depot is fully connected to all the other nodes
        W[0, :] = 1
        W[:, 0] = 1

        for idx in range(num_nodes):
            W[idx][knns[idx]] = 1
            W = W.T
            W[idx][knns[idx]] = 1

        np.fill_diagonal(W, 0)

        W_val *= W
        return W.astype(int), W_val

    def get_time_based_distance_matrix(self, W):
        return (W / self.speed) * 60

    def bss_graph_gen(self):
        self.gen_instance()
        self.W, self.W_val = self.adjacenct_gen(self.num_nodes, self.num_neighbors, self.static)
        self.W_weighted = torch.tensor(np.multiply(self.W_val, self.W))
        self.A = sparse.csr_matrix(self.W_weighted)
        self.g = nx.from_numpy_matrix(np.matrix(self.W), create_using=nx.Graph)
        self.g_weighted = nx.from_numpy_matrix(np.matrix(self.W_weighted), create_using=nx.Graph)
        self.g.edges(data=True)

    def nodes(self):

        return nx.number_of_nodes(self.g)

    def edges(self):

        return self.g.edges()

    def neighbors(self, node):

        return nx.all_neighbors(self.g, node)

    def average_neighbor_degree(self, node):

        return nx.average_neighbor_degree(self.g, nodes=node)

    def adj(self):

        return nx.adjacency_matrix(self.g)

    def get_state(self):
        edge_index, edge_weight = from_scipy_sparse_matrix(self.A)
        edge_index = edge_index.long()
        edge_weight = edge_weight.float()
        return edge_index,edge_weight

    def get_demands(self):
        """ Gets random demand vector that has zero sum. """

        # randomly sample demands
        demands = self.rng.randint(1, self.max_demand, self.num_nodes)
        demands *= self.rng.choice([-1, 1], self.num_nodes)  # exclude 0

        # zero demand at depot
        demands[0] = 0

        # adjust demands until they sum to zero.
        while True:

            if demands.sum() == 0:
                return demands

            demand_sum_pos = demands.sum() > 0

            idx = self.rng.randint(1, demands.shape[0])

            if demands[idx] < 0:
                if demand_sum_pos:
                    if demands[idx] == - self.max_demand:
                        continue
                    demands[idx] -= 1
                else:
                    if demands[idx] == -1:  # case for over -1 to 1
                        demands[idx] += 1
                    demands[idx] += 1

            elif demands[idx] > 0:
                if demand_sum_pos:
                    if demands[idx] == 1:  # case for over 1 to -1
                        demands[idx] -= 1
                    demands[idx] -= 1
                else:
                    if demands[idx] == self.max_demand:
                        continue
                    demands[idx] += 1

        return demands


# Toy Case Test
def test():
    g = Graph(
        num_nodes=10,
        k_nn=4,
        num_vehicles=3,
        penalty_cost_demand=1,
        penalty_cost_time=1,
        speed=30,
        time_limit=120)
    nx.draw(g.g, with_labels=True)
    edge_index, edge_weight = g.get_state()
    print(edge_index)


if __name__ == "__main__":
    test()
