import numpy as np
import random
import os
import logging
import models
from utils.config import load_model_config
import torch
from collections import namedtuple, deque

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

"""
Contains the definition of the agent that will run in an
environment.
"""

BATCH_SIZE = 64  # batch size of sampling process from buffer

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'w_weight'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQAgent:

    def __init__(self, model, lr):

        self.embed_dim = 64
        self.model_name = model

        self.k = 20
        self.alpha = 0.1
        self.gamma = 1  # 0.99
        self.lambd = 0.

        self.epsilon_ = 0.8
        self.epsilon_min = 0.02
        self.discount_factor = 0.999
        self.neg_inf = torch.tensor(-1000)
        self.T = 5
        self.t = 1

        self.TARGET_NETWORK_REPLACE_FREQ = 10  # How frequently target netowrk updates
        self.MEMORY_CAPACITY = 2000  # The capacity of experience replay buffer

        if self.model_name == 'GCN_QN_1':
            args_init = load_model_config()[self.model_name]
            self.policy_net, self.target_net = models.GCN_QN_1(**args_init), models.GCN_QN_1(**args_init)

        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # # ----Define the memory (or the buffer), allocate some space to it. The number
        self.memory = ReplayMemory(self.MEMORY_CAPACITY)

        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # ------Define the loss function-----#
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def choose_action(self, state, adj_weighted, back_depot):
        # Clone the dynamic variable so we don't mess up graph
        observation = state[0]
        loads = state[1]
        demands = state[2]
        cur_node = (state[3] == 1).nonzero().item()
        last_node = (state[4] == 1).nonzero().item()

        # all_loads = loads.clone()
        # all_demands = demands.clone()
        # assert(all_demands.sum() == all_loads[0] ) # TODO check system is balanced
        # Otherwise, we can choose to go anywhere where demand is > 0
        # mask = all_demands.ne(0)  # * observation.reshape(20).ne(1)
        # mask *= all_demands.lt(all_loads)
        # mask *= (-all_demands).lt(20 - all_loads)

        nbr_nodes = (adj_weighted[cur_node] != 0).int()
        uncovered_nodes = (observation[:] == 0).int()
        mask = nbr_nodes * uncovered_nodes
        mask[0] = 1  # depot is always available unless last visit
        mask[last_node] = 0  # mask out visited node

        # Time limit constraint: all vehicles must start at the depot and return to the depot within a limited time.
        if back_depot and (last_node != 0):  # TODO: hardcore  return to depot after traveling 120 min at 30km/h
            action = torch.tensor([0])  # return to depot

        elif self.epsilon_ > torch.rand(1):
            if (nbr_nodes * mask == 1).any():  # if there is node neighbor to go
                action = torch.tensor([np.random.choice(np.where(nbr_nodes * mask == 1)[0])])
            else:
                action = torch.tensor([0])  # return to depot

        else:
            q_a = self.policy_net(state, adj_weighted).detach().clone()
            action = torch.argmax(q_a[0, :, 0] + (1 - mask) * self.neg_inf).reshape(1)

            # q_a = self.model(observation, self.graphs[self.games].A).detach().clone()
            # edge_index, edge_weight = from_scipy_sparse_matrix(self.graphs[self.games].A)
            # q_a = self.model(observation.squeeze(0), edge_index, edge_weight.float()).detach().clone()
            # q_a = q_a.unsqueeze(0)

        return action

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network

        # update the target network every fixed steps
        if self.learn_step_counter % self.TARGET_NETWORK_REPLACE_FREQ == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

        # # Determine the Sampled batch from buffer
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        b_s = torch.stack(batch.state).float()  # torch.Size([1, 10, 6])
        b_a = torch.stack(batch.action)
        b_r = torch.stack(batch.reward)
        b_s_ = torch.stack(batch.next_state).float()
        b_w_weight = torch.stack(batch.w_weight).float()

        # calculate the Q value of state-action pair
        a_idx = b_a.unsqueeze(1).repeat(1, b_w_weight.shape[1], 1)
        q_eval = self.policy_net(b_s, b_w_weight).gather(1, a_idx)[:, 0, :]
        # calculate the q value of next state
        q_next = self.target_net(b_s_, b_w_weight).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = (b_r + self.gamma * q_next.max(1)[0].view(BATCH_SIZE, 1)).float()  # (batch_size, 1)
        loss = self.criterion(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step

        # epsilon decay rule
        if self.epsilon_ > self.epsilon_min:
            self.epsilon_ *= self.discount_factor

    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), cwd + '/model.pt')


Agent = DQAgent
