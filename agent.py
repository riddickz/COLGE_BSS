import numpy as np
import random
import os
import logging
import models
from utils.config import load_model_config
import torch
from collections import namedtuple, deque
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.nn.utils.rnn import pad_sequence
import copy

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

"""
Contains the definition of the agent that will run in an
environment.
"""

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'adj'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQAgent:

    def __init__(self, model, lr):
        self.model_name = model
        self.gamma = 0.99  # 0.99
        self.epsilon_ = 0.8
        self.epsilon_min = 0.02
        self.discount_factor = 0.999
        self.neg_inf = torch.tensor(-1000)

        self.target_net_replace_freq = 20  # How frequently target netowrk updates
        self.mem_capacity = 4000 # capacity of experience replay buffer
        self.batch_size = 64  # batch size of sampling process from buffer

        # if self.model_name == 'GCN_QN_1':
        #     args_init = load_model_config()[self.model_name]
        #     self.policy_net = models.GCN_QN_1(**args_init)
        # elif self.model_name == 'GCN_QN_1':
        #      self.policy_net = models.GCN2_Net(input_channels=8, output_channels=1, hidden_channels=8, num_layers=10, alpha=0.1,
        #                  theta=0.5, shared_weights=True, dropout=0.6)
        # elif self.model_name == 'GCN_Naive':
        #      self.policy_net = models.GCN_Naive(c_in=8, c_out=1, c_hidden=8)

        self.policy_net = models.GATv2(in_features=8, n_hidden=8, n_classes=1, n_heads=1, dropout=0.1, share_weights=False)
        self.target_net = copy.deepcopy(self.policy_net)

        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # # ----Define the memory (or the buffer), allocate some space to it. The number
        self.memory = ReplayMemory(self.mem_capacity)

        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # ------Define the loss function-----#
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def choose_action(self, state, adj, back_depot):
        # Clone the dynamic variable so we don't mess up graph
        observation = state[0].int()
        loads = state[1]
        demands = state[2]
        cur_node = (state[3] == 1).nonzero().item()
        last_node = (state[4] == 1).nonzero().item()

        nbr_nodes = (adj[cur_node] > 0).int()
        uncovered_nodes = (observation[:] == 0).int()
        mask = nbr_nodes * uncovered_nodes
        mask[0] = 1  # depot is always available unless last visit
        mask[last_node] = 0
        mask[cur_node] =  0 # mask out visited node

        # Time limit constraint: all vehicles must start at the depot and return to the depot within a limited time.
        if back_depot and (last_node > 0) and (cur_node > 0):
            action = torch.tensor([0])

        elif self.epsilon_ > torch.rand(1):
            if (nbr_nodes * mask == 1).any():  # if there is node neighbor to go
                action = torch.tensor([np.random.choice(np.where(nbr_nodes * mask == 1)[0])])
            else:
                action = torch.tensor([0])  # return to depot

        else:
            q_a = self.policy_net(state.T.unsqueeze(0), adj.unsqueeze(0)).detach().clone()
            # q_a = self.policy_net(state.T, edge_index).detach().clone()

            action = torch.argmax(q_a[:,0] + (1 - mask) * self.neg_inf).reshape(1)

        return action

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network

        # update the target network every fixed steps
        if self.learn_step_counter % self.target_net_replace_freq == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

        # # Determine the Sampled batch from buffer
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        b_s = torch.stack(batch.state).permute(0,2,1).float()  # torch.Size([1, 10, 6])
        b_a = torch.stack(batch.action)
        b_r = torch.stack(batch.reward)
        b_s_ = torch.stack(batch.next_state).permute(0,2,1).float()
        b_adj = torch.stack(batch.adj).float()
        # b_w_weight = torch.stack(batch.w_weight).float()
        # b_edge_index = pad_sequence(list(batch.edge_index), padding_value=0).permute(1,2,0).long()
        # b_edge_weight = pad_sequence(list(batch.edge_weight), padding_value=0).T.float()

        # calculate the Q value of state-action pair
        a_idx = b_a.unsqueeze(1).repeat(1, b_adj.shape[1], 1)
        q_eval = self.policy_net(b_s,b_adj).gather(1, a_idx)[:, 0, :]
        # calculate the q value of next state
        q_next = self.target_net(b_s_,b_adj).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = (b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)).float()  # (batch_size, 1)

        # q_eval = torch.zeros(BATCH_SIZE)
        # q_target = torch.zeros(BATCH_SIZE)
        # for i in range(BATCH_SIZE):
        #     q_eval[i] = self.policy_net(batch.state[i].T, batch.edge_index[i])[batch.action[i]]
        #     q_next = self.target_net(batch.next_state[i].T, batch.edge_index[i]).detach()  # detach from computational graph, don't back propagate
        #     q_target[i] = (batch.reward[i] + self.gamma * q_next.max()).float()

        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step

        # epsilon decay rule
        if self.epsilon_ > self.epsilon_min:
            self.epsilon_ *= self.discount_factor

    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.policy_net.state_dict(), cwd + '/model.pt')


Agent = DQAgent
