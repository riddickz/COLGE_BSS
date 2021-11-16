import numpy as np
import networkx as nx
import collections
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os
import torch
from scipy import sparse

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# seed = np.random.seed(120)

class Graph:
    def __init__(self, 
        cur_n, 
        k_nn, 
        num_vehicles,
        penalty_cost_demand,
        penalty_cost_time, 
        speed,
        time_limit,
        max_load=20, 
        max_demand=9, 
        area = 10, 
        seed=None):

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        self.seed = seed
        self.num_nodes = cur_n
        self.num_neighbors = k_nn
        self.max_load = max_load
        self.max_demand = max_demand
        self.area = area #km
        self.num_vehicles = num_vehicles
        self.penalty_cost_demand = penalty_cost_demand
        self.penalty_cost_time = penalty_cost_time
        self.speed = speed
        self.time_limit = time_limit
        self.bss_graph_gen()

    def gen_instance(self):  # Generate random instance
        seed = np.random.randint(123456789)
        np.random.seed(seed)

        locations = np.random.rand(self.num_nodes, 2) * self.area  # node num with (dimension) coordinates in [0,1]
        # pca = PCA(n_components=2)  # center & rotate coordinates
        # locations = pca.fit_transform(coords)

        demands = np.random.randint(1, self.max_demand, self.num_nodes) * np.random.choice([-1, 1],
                                                                                           self.num_nodes)  # exclude 0
        demands[0] = -sum(demands[1:])  # depot
        demands = torch.tensor(demands)

        loads = torch.zeros(self.num_nodes)

        self.static = torch.tensor(locations)
        self.observation = torch.zeros(self.num_nodes)
        self.dynamic = torch.stack((self.observation, loads, demands), dim=1)


    def adjacenct_gen(self, num_nodes, num_neighbors, coords):
        assert num_neighbors < num_nodes
        # if num_nodes <= 20:
        #     num_neighbors = 7
        #     # num_neighbors = num_nodes - 1 # fully connected
        # else:
        #     num_neighbors = 15
        #     # num_neighbors = int(np.random.uniform(low=0.9, high=1) * 8)

        # add KNN edges with random K
        W_val = squareform(pdist(coords, metric='euclidean'))

        W = np.zeros((num_nodes, num_nodes))
        knns = np.argpartition(W_val, kth=num_neighbors, axis=-1)[:, num_neighbors::-1]

        # depot is fully connected to all the other nodes
        W[0,:] = 1
        W[:,0] = 1

        for idx in range(num_nodes):
            W[idx][knns[idx]] = 1
            W = W.T
            W[idx][knns[idx]] = 1

        np.fill_diagonal(W, 0)

        W_val *= W
        return W.astype(int), W_val

    def bss_graph_gen(self):
        self.gen_instance()
        self.W, self.W_val = self.adjacenct_gen(self.num_nodes, self.num_neighbors, self.static)
        self.W_weighted = torch.tensor(np.multiply(self.W_val, self.W))
        self.A = sparse.csr_matrix(self.W_weighted)
        self.g = nx.from_numpy_matrix(np.matrix(self.W_weighted), create_using=nx.Graph)
        self.g.edges(data=True)

    def nodes(self):

        return nx.number_of_nodes(self.g)

    def edges(self):

        return self.g.edges()

    def neighbors(self, node):

        return nx.all_neighbors(self.g,node)

    def average_neighbor_degree(self, node):

        return nx.average_neighbor_degree(self.g, nodes=node)

    def adj(self):

        return nx.adjacency_matrix(self.g)

# Toy Case Test
g = Graph(cur_n=10, k_nn=4, penalty_cost=1, time_limit=120)
# G = nx.from_numpy_array(g.W_weighted)
nx.draw(g.g, with_labels=True)
plt.show()
pass

# layout = nx.spring_layout(G)
# nx.draw(G, layout)
# nx.draw_networkx_edge_labels(G, pos=layout)
# plt.show()