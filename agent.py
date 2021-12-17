import numpy as np
import random
import os
import logging
import models
from utils.config import load_model_config
import torch
import copy
from utils.vis import plot_grad_flow,count_parameters,timestamp
from replay_buffer import ReplayMemory, ReplayBuffer
from labml_helpers.schedule import Piecewise
from torch.optim import lr_scheduler

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

"""
Contains the definition of the agent that will run in an
environment.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DQAgent:

    def __init__(self, model, lr,bs, replace_freq, num_node = 10):
        self.model_name = model
        self.gamma = .99  # 0.99
        self.epsilon_ = 0.95 #eps
        self.epsilon_min = 0.01 #0.05
        self.discount_factor = 0.99995
        self.neg_inf = -100000
        self.num_node = num_node
        self.num_ft = 14

        self.target_net_replace_freq = replace_freq  # How frequently target netowrk updates
        # self.mem_capacity = 30000 # capacity of experience replay buffer ,100000
        self.mem_capacity = 2 ** 15 #  must be a power of 2.
        self.batch_size = bs  # batch size of sampling process from buffer

        # elif self.model_name == 'GCN_Naive':
        #      self.policy_net = models.GCN_Naive(c_in=8, c_out=1, c_hidden=8)
        self.policy_net = models.GATv2(in_features=self.num_ft, n_hidden=128, n_classes=1, n_node=self.num_node , n_heads=1, dropout=0.1, share_weights=False).to(device)
        self.target_net = copy.deepcopy(self.policy_net).to(device)

        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # ------- Define the memory (or the buffer)------#
        # self.memory = ReplayMemory(self.mem_capacity)
        # beta for replay buffer as a function of updates
        self.prioritized_replay_beta = Piecewise(
            [
                (0, 0.),
                (5*3000, 1)
            ], outside_value=1)
        self.prioritized_replay_alpha = 0.5
        # Replay buffer with α=0.6. Capacity of the replay buffer must be a power of 2.
        self.replay_buffer = ReplayBuffer(self.mem_capacity, self.prioritized_replay_alpha, self.num_node, self.num_ft )

        # ------- Define the optimizer------#
        # self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay= 0.01)
        # self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=lr, momentum= 0.9, weight_decay= 0.01)
        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=lr, momentum= 0.9, weight_decay= 0.01)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)

        # ------Define the loss function-----#
        self.criterion = torch.nn.SmoothL1Loss(reduction='none')

    def choose_action(self, state, adj, mask):
        pr = torch.rand(1)

        if self.epsilon_ > pr:
            mask = mask.detach().cpu().numpy()
            q_a = None
            action = torch.tensor([np.random.choice(np.where(mask[0] == 1)[0])]) # randomly choose any unvisited node + depot
        else:
            q_a = self.policy_net(state.T.unsqueeze(0), adj.unsqueeze(0), mask=None).detach().clone().to(device)
            action = torch.argmax(q_a[0,:,0] + (1 - mask) * self.neg_inf).reshape(1)

        return action.to(device), q_a

    def learn(self,iter_count):
        # sampling batch of experiences, update parameters of target network

        # update the target network every fixed steps
        if self.learn_step_counter % self.target_net_replace_freq == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

        # Determine the Sampled batch from buffer
        # transitions = self.memory.sample(self.batch_size)
        # batch = Transition(*zip(*transitions))
        beta = self.prioritized_replay_beta(iter_count)
        transitions = self.replay_buffer.sample(self.batch_size, beta=beta)

        b_s =torch.tensor(transitions["obs"]).permute(0,2,1).float().to(device) # torch.Size([1, 10, 6])
        b_a = torch.tensor(transitions['action']).reshape(self.batch_size,1)
        b_r = torch.tensor(transitions['reward']).reshape(self.batch_size,1)
        b_s_ = torch.tensor(transitions['next_obs']).permute(0,2,1).float().to(device)
        b_adj = torch.tensor(transitions['adj']).float().to(device)
        b_weight = torch.tensor(transitions['weights']).reshape(self.batch_size,1,1).float().to(device)

        # calculate the Q value of state-action pair
        a_idx = b_a.unsqueeze(-1)
        q_eval = self.policy_net(b_s,b_adj,mask = None).gather(1, a_idx)

        # double-DQN
        with torch.no_grad():
            # select the maximum q value
            best_a = self.policy_net(b_s, b_adj, mask = None).argmax(1).unsqueeze(-1)
            best_q_next = self.target_net(b_s_, b_adj, mask = None).gather(1, best_a).to(device)

            b_r = torch.clamp(b_r, min=-1, max=1).to(device) # reward clipped within [−1, 1] for stability
            q_target = (b_r.unsqueeze(-1) + self.gamma * best_q_next).float().to(device)  # (batch_size, 1)
            td_errors = q_eval - q_target


        losses = self.criterion(q_eval, q_target)
        loss = torch.mean(b_weight * losses)

        if loss.abs() > 0.9:
            print("WARNING: HIGH LOSS" )
            print(loss)

        loss = torch.clamp(loss, min=-1, max=1) # TD error clipped within [−1, 1] for stability

        # Calculate priorities for replay buffer
        new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6

        # Update replay buffer priorities
        self.replay_buffer.update_priorities(transitions['indexes'], new_priorities)



        self.optimizer.zero_grad()  # reset the gradient to zero

        loss.backward(retain_graph=True)

        # torch.nn.utils.clip_grad.clip_grad_norm_(self.policy_net.parameters(), 10)
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()  # execute back propagation for one step

        # epsilon decay rule
        if self.epsilon_ > self.epsilon_min:
            self.epsilon_ *= self.discount_factor

        return loss, self.epsilon_

    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.policy_net.state_dict(), cwd + '/trained_models/model_{}.pt'.format(timestamp()))

    def load_model(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))

    def cuda(self):
        self.policy_net = self.policy_net.cuda()
        self.target_net = self.target_net.cuda()

    def cpu(self):
        self.policy_net = self.policy_net.cpu()
        self.target_net = self.target_net.cpu()

Agent = DQAgent
