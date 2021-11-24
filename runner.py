"""
This is the machinnery that runs your agent in an environment.

"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import agent
from utils.vis import plot_reward


class Runner:
    def __init__(self, environment, agent, verbose=False, render=False):
        self.env = environment
        self.agent = agent
        self.verbose = verbose
        self.render_on = render

    def train(self, g, max_episode, max_iter):
        cumul_reward = []

        for i_episode in range(max_episode):  # TODO: change hardcode
            s, adj_mat = self.env.reset(g)
            back_depot = False
            ep_r = 0

            for i in range(0, max_iter):
                a = self.agent.choose_action(s, adj_mat, back_depot)

                # obtain the reward and next state and some other information
                s_, r, done, info = self.env.step(a)
                back_depot = info[3]

                # Store the transition in memory
                self.agent.memory.push(s, a, r, s_, adj_mat)
                self.agent.memory_counter += 1

                ep_r += r.item()

                # if the experience replay buffer is filled, DQN begins to learn or update its parameters
                if self.agent.memory_counter > self.agent.mem_capacity:
                    self.agent.learn()
                    if done:
                        print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

                if done:
                    # if game is over, then skip the while loop.
                    print(" ->    Terminal event: episodic rewards = {}".format(ep_r))
                    break

                # use next state to update the current state.
                s = s_
            if self.render_on:
                self.env.render()
        return cumul_reward

    def train_loop(self, games, max_epoch, max_episode=30, max_iter=1000):
        cumul_reward_list = []

        # Start training
        print("\nCollecting experience...")
        for epoch_ in range(max_epoch):
            print(" -> epoch : " + str(epoch_))
            for g in range(1, games + 1):
                print(" -> games : " + str(g))
                cumul_reward = self.train(g, max_episode, max_iter)
                cumul_reward_list.extend(cumul_reward)

                if self.verbose:
                    print(" <=> Finished game number: {} <=>\n".format(g))
