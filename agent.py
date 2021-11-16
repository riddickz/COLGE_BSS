import numpy as np
import random
import time
import os
import logging
import models
import tgcn
from utils.config import load_model_config
from scipy import sparse

import torch.nn.functional as F
import torch
from torch_geometric.utils import from_scipy_sparse_matrix

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

"""
Contains the definition of the agent that will run in an
environment.
"""


class DQAgent:

    def __init__(self, graph, model, lr, bs, n_step):

        self.graphs = graph
        self.embed_dim = 64
        self.model_name = model

        self.k = 20
        self.alpha = 0.1
        self.gamma = 0.99
        self.lambd = 0.
        self.n_step = n_step

        self.epsilon_ = 0.8
        self.epsilon_min = 0.02
        self.discount_factor = 0.999
        # self.eps_end=0.02
        # self.eps_start=1
        # self.eps_step=20000
        self.t = 1
        self.memory = []
        self.memory_n = []
        self.minibatch_length = bs
        self.neg_inf = torch.tensor(-1000)
        self.num_vehicle = 2

        if self.model_name == 'S2V_QN_1':

            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_1(**args_init)

        elif self.model_name == 'S2V_QN_2':
            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_2(**args_init)


        elif self.model_name == 'GCN_QN_1':

            args_init = load_model_config()[self.model_name]
            self.model = models.GCN_QN_1(**args_init)

        elif self.model_name == "TGCN":
            self.model = tgcn.TGCN(adj=self.adj_weighted, hidden_dim=64)

        elif self.model_name == "RecurrentGCN":
            self.model = models.RecurrentGCN(node_features=3)

        elif self.model_name == 'LINE_QN':

            args_init = load_model_config()[self.model_name]
            self.model = models.LINE_QN(**args_init)

        elif self.model_name == 'W2V_QN':

            args_init = load_model_config()[self.model_name]
            self.model = models.W2V_QN(G=self.graphs[self.games], **args_init)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.T = 5

        self.t = 1

    """
    p : embedding dimension

    """

    def reset(self, g):

        self.games = g

        if (len(self.memory_n) != 0) and (len(self.memory_n) % 300000 == 0):
            self.memory_n = random.sample(self.memory_n, 120000)

        self.graph = self.graphs[self.games]
        self.nodes = self.graph.nodes()
        self.adj = torch.tensor(self.graph.W).unsqueeze(dim=0).float()
        self.adj_weighted = torch.tensor(self.graph.W_weighted).unsqueeze(dim=0).float()
        self.last_action = 0
        self.last_observation = torch.zeros(1, self.nodes, 3, dtype=torch.float)
        self.last_reward = -0.01
        self.last_done = 0
        self.iter = 1
        self.cur_node = 0
        self.trip_len = 0 # for time limit constr

    def act(self, observation, dynamic, tour_length):
        nbr_nodes = torch.tensor(self.graph.W[self.cur_node])
        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[0].clone()
        all_demands = dynamic[1].clone()
        assert(all_demands.sum() == all_loads[0] ) # check system is balanced

        # Otherwise, we can choose to go anywhere where demand is > 0
        # new_mask = all_demands.ne(0)  # * observation.reshape(20).ne(1)
        # new_mask *= all_demands.lt(all_loads)
        # new_mask *= (-all_demands).lt(20 - all_loads)

        # # avoid isolation
        # if (next_locs * new_mask).eq(1).any():
        #     next_locs *= new_mask
        #
        # avoid repeating nodes
        nbr_nodes[self.last_action] = 0

        # nodes with demand fully covered
        nodes_uncovered = (observation[0, :, 0] == 0)

        node_visited = torch.zeros_like(nbr_nodes)
        node_visited[0] = 0 # depot is always available unless last visit
        node_visited[self.last_action] = 1

        mask = nbr_nodes * nodes_uncovered * (1 - node_visited)

        chosen_idx = 0

        # Time limit constraint
        # All vehicles must start at the depot and return to the depot within a limited time.
        if (tour_length % 60 == 0) and (self.last_action != 0):  # TODO  return to depot after traveling 120 min at 30km/h
            return chosen_idx # return to depot

        elif self.epsilon_ > torch.rand(1):
            if (nbr_nodes* mask == 1).any():
                chosen_idx = np.random.choice(np.where(nbr_nodes* mask == 1)[0])

        else:
            q_a = self.model(observation, self.adj_weighted).detach().clone()
            # q_a = self.model(observation, self.graphs[self.games].A).detach().clone()
            # edge_index, edge_weight = from_scipy_sparse_matrix(self.graphs[self.games].A)
            # q_a = self.model(observation.squeeze(0), edge_index, edge_weight.float()).detach().clone()
            # q_a = q_a.unsqueeze(0)
            chosen_idx = torch.argmax(q_a[0, :, 0] + (1- mask) * self.neg_inf ).item()
            try:
                assert chosen_idx != self.last_action
            except:
                print(chosen_idx, self.last_action)

        self.cur_node = chosen_idx
        self.trip_len += 1

        return chosen_idx

    def reward(self, observation, action, reward, done):

        if len(self.memory_n) > self.minibatch_length + self.n_step:  # or self.games > 2:

            (last_observation_tens, action_tens, reward_tens, observation_tens, done_tens,adj_tens) = self.get_sample()
            # observation_tens = observation_tens[:, :, 0].unsqueeze(2)
            target = reward_tens + self.gamma *(1-done_tens)*torch.max(self.model(observation_tens, adj_tens) + observation_tens[:, :, 0].unsqueeze(2) * (-1e5), dim=1)[0]
            target_f = self.model(last_observation_tens, adj_tens)
            target_p = target_f.clone()
            target_f[range(self.minibatch_length), action_tens, :] = target
            loss = self.criterion(target_p, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(self.t, loss)

            # self.epsilon = self.eps_end + max(0., (self.eps_start- self.eps_end) * (self.eps_step - self.t) / self.eps_step)
            if self.epsilon_ > self.epsilon_min:
                self.epsilon_ *= self.discount_factor
        if self.iter > 1:
            self.remember(self.last_observation, self.last_action, self.last_reward, observation.clone(),
                          self.last_done * 1)

        if done & self.iter > self.n_step:
            self.remember_n(False)
            new_observation = observation.clone()
            new_observation[:, action, :] = 1
            self.remember(observation, action, reward, new_observation, done * 1)

        if self.iter > self.n_step:
            self.remember_n(done)
        self.iter += 1
        self.t += 1
        self.last_action = action
        self.last_observation = observation.clone()
        self.last_reward = reward
        self.last_done = done

    def get_sample(self):

        minibatch = random.sample(self.memory_n, self.minibatch_length - 1)
        minibatch.append(self.memory_n[-1])
        last_observation_tens = minibatch[0][0]
        action_tens = torch.Tensor([minibatch[0][1]]).type(torch.LongTensor)
        reward_tens = torch.Tensor([[minibatch[0][2]]])
        observation_tens = minibatch[0][3]
        done_tens = torch.Tensor([[minibatch[0][4]]])
        adj_tens = self.graphs[minibatch[0][5]].adj().todense()
        adj_tens = torch.from_numpy(np.expand_dims(adj_tens.astype(int), axis=0)).type(torch.FloatTensor)

        for last_observation_, action_, reward_, observation_, done_, games_ in minibatch[-self.minibatch_length + 1:]:
            last_observation_tens = torch.cat((last_observation_tens, last_observation_))
            action_tens = torch.cat((action_tens, torch.Tensor([action_]).type(torch.LongTensor)))
            reward_tens = torch.cat((reward_tens, torch.Tensor([[reward_]])))
            observation_tens = torch.cat((observation_tens, observation_))
            done_tens = torch.cat((done_tens, torch.Tensor([[done_]])))
            adj_ = self.graphs[games_].adj().todense()
            adj = torch.from_numpy(np.expand_dims(adj_.astype(int), axis=0)).type(torch.FloatTensor)
            adj_tens = torch.cat((adj_tens, adj))

        return (last_observation_tens, action_tens, reward_tens, observation_tens, done_tens, adj_tens)

    def remember(self, last_observation, last_action, last_reward, observation, done):
        self.memory.append((last_observation, last_action, last_reward, observation, done, self.games))

    def remember_n(self, done):

        if not done:
            step_init = self.memory[-self.n_step]
            cum_reward = step_init[2]
            for step in range(1, self.n_step):
                cum_reward += self.memory[-step][2]
            self.memory_n.append(
                (step_init[0], step_init[1], cum_reward, self.memory[-1][-3], self.memory[-1][-2], self.memory[-1][-1]))

        else:
            for i in range(1, self.n_step):
                step_init = self.memory[-self.n_step + i]
                cum_reward = step_init[2]
                for step in range(1, self.n_step - i):
                    cum_reward += self.memory[-step][2]
                if i == self.n_step - 1:
                    self.memory_n.append(
                        (step_init[0], step_init[1], cum_reward, self.memory[-1][-3], False, self.memory[-1][-1]))

                else:
                    self.memory_n.append(
                        (step_init[0], step_init[1], cum_reward, self.memory[-1][-3], False, self.memory[-1][-1]))

    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), cwd + '/model.pt')


Agent = DQAgent
