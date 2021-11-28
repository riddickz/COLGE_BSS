import numpy as np
import random
import os
import logging
import models
from utils.config import load_model_config
import torch
from collections import namedtuple, deque
import copy
from utils.vis import plot_grad_flow,count_parameters,timestamp

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
                        ('state', 'action', 'reward', 'next_state', 'adj', 'mask'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_list(self, size):
        x = []
        for _ in range(size):
            i = random.randrange(len(self.memory) - 5)
            x.append(self.memory[i])
        return x

    def __len__(self):
        return len(self.memory)

class DQAgent:

    def __init__(self, model, lr,bs, replace_freq):
        self.model_name = model
        self.gamma = .99  # 0.99
        self.epsilon_ = 0.8
        self.epsilon_min = 0.05
        self.discount_factor = 0.99
        self.neg_inf = -1000

        self.target_net_replace_freq = replace_freq  # How frequently target netowrk updates
        self.mem_capacity = 20000 # capacity of experience replay buffer
        self.batch_size = bs  # batch size of sampling process from buffer

        # elif self.model_name == 'GCN_Naive':
        #      self.policy_net = models.GCN_Naive(c_in=8, c_out=1, c_hidden=8)

        self.policy_net = models.GATv2(in_features=8, n_hidden=64, n_classes=1, n_heads=1, dropout=0.0, share_weights=False).to(device)
        self.target_net = copy.deepcopy(self.policy_net).to(device)

        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # # ----Define the memory (or the buffer), allocate some space to it. The number
        self.memory = ReplayMemory(self.mem_capacity)
        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay= 0.01)

        # ------Define the loss function-----#
        self.criterion = torch.nn.SmoothL1Loss()

    def choose_action(self, state, adj, mask):
        pr = torch.rand(1)
        if self.epsilon_ > pr:
            mask = mask.detach().cpu().numpy()
            action = torch.tensor([np.random.choice(np.where(mask[0] == 1)[0])]) # randomly choose any unvisited node + depot
                # action = torch.tensor([0])  # return to depot
        else:
            q_a = self.policy_net(state.T.unsqueeze(0), adj.unsqueeze(0), mask).detach().clone().to(device)
            action = torch.argmax(q_a[0,:,0] + (1 - mask) * self.neg_inf).reshape(1)
        return action.to(device)

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network

        # update the target network every fixed steps
        if self.learn_step_counter % self.target_net_replace_freq == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

        # Determine the Sampled batch from buffer
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        b_s = torch.stack(batch.state).permute(0,2,1).float() # torch.Size([1, 10, 6])
        b_a = torch.stack(batch.action)
        b_r = torch.stack(batch.reward)
        b_s_ = torch.stack(batch.next_state).permute(0,2,1).float().to(device)
        b_adj = torch.stack(batch.adj).float().to(device)
        b_mask = torch.cat(batch.mask)

        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         b_s_)), device=device, dtype=torch.bool)
        #
        # non_final_next_states = torch.stack([s for s in b_s_
        #                                    if s is not None])

        # calculate the Q value of state-action pair
        a_idx = b_a.unsqueeze(1).repeat(1, b_adj.shape[1], 1)
        q_eval = self.policy_net(b_s,b_adj,b_mask).gather(1, a_idx)[:, 0, :]

        # calculate the q value of next state
        # q_next = torch.zeros(b_s_.size(0),b_s_.size(1),1, device=device)
        # q_next[non_final_mask]  = self.target_net(non_final_next_states,b_adj).detach()  # detach from computational graph, don't back propagate
        q_next = self.target_net(b_s_,b_adj,b_mask).detach().to(device)  # detach from computational graph, don't back propagate

        # select the maximum q value
        b_r = torch.clamp(b_r, min=-1, max=1).to(device)
        q_target = (b_r.reshape(-1,1) + self.gamma * q_next.max(1)[0]).float().to(device)  # (batch_size, 1)

        loss = self.criterion(q_eval, q_target)
        if loss > 0.01:
            print("WARNING: HIGH LOSS" )
            print(loss)

        self.optimizer.zero_grad()  # reset the gradient to zero

        loss.backward(retain_graph=True)

        # torch.nn.utils.clip_grad.clip_grad_norm_(self.policy_net.parameters(), 10)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()  # execute back propagation for one step

        # epsilon decay rule
        if self.epsilon_ > self.epsilon_min:
            self.epsilon_ *= self.discount_factor

        return loss, self.epsilon_

    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.policy_net.state_dict(), cwd + '/model_{}.pt'.format(timestamp()))

    def load_model(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))


Agent = DQAgent
