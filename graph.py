import numpy as np
import networkx as nx
import collections
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# seed = np.random.seed(120)

class Graph:
    def __init__(self, graph_type, cur_n, p=None, m=None, max_load=20, max_demand=9, seed=None):

        if graph_type == 'erdos_renyi':
            self.g = nx.erdos_renyi_graph(n=cur_n, p=p, seed=seed)
        elif graph_type == 'powerlaw':
            self.g = nx.powerlaw_cluster_graph(n=cur_n, m=m, p=p, seed=seed)
        elif graph_type == 'barabasi_albert':
            self.g = nx.barabasi_albert_graph(n=cur_n, m=m, seed=seed)
        elif graph_type =='gnp_random_graph':
            self.g = nx.gnp_random_graph(n=cur_n, p=p, seed=seed)

        elif graph_type =='bss':
            if max_load < max_demand:
                raise ValueError(':param max_load: must be > max_demand')

            self.seed = seed
            self.max_nodes = cur_n
            self.max_load = max_load
            self.max_demand = max_demand
            self.static = None
            self.dynamic = None
            self.W, self.W_weighted = self.bss_graph_gen()
            self.g = nx.from_numpy_array(self.W_weighted)
            self.g.edges(data=True)

    def gen_instance(self):  # Generate random instance
        seed = np.random.randint(123456789)
        np.random.seed(seed)

        coords = np.random.rand(self.max_nodes, 2)  # node num with (dimension) coordinates in [0,1]
        pca = PCA(n_components=2)  # center & rotate coordinates
        locations = pca.fit_transform(coords)

        demands = np.random.randint(1, self.max_demand, self.max_nodes) * np.random.choice([-1, 1],
                                                                                           self.max_nodes)  # exclude 0
        demands[0] = -sum(demands[1:])  # depot
        demands = torch.tensor(demands)

        loads = torch.zeros(self.max_nodes)

        self.static = torch.tensor(locations)
        self.dynamic = torch.stack((loads, demands), dim=0)

    def adjacenct_gen(self, num_nodes, coords):

        if num_nodes <= 20:
            num_neighbors = 7
            # num_neighbors = num_nodes - 1 # fully connected
        else:
            num_neighbors = 15
            # num_neighbors = int(np.random.uniform(low=0.9, high=1) * 8)

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

        W, W_val = self.adjacenct_gen(self.max_nodes, self.static)
        return W, np.multiply(W_val, W)

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

# # Toy Case Test
# g = Graph(graph_type="bss")
# # G = nx.from_numpy_array(g.W_weighted)
# nx.draw(g.g, with_labels=True)
# plt.show()
# pass