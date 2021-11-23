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

    def train(self, g, max_iter):
        cumul_reward = []

        for i_episode in range(100):  # TODO: change hardcode
            # play episodes of BSS game on same graph with different demand
            s, w_weighted, edge_index, edge_weight = self.env.reset(g)

            back_depot = False
            ep_r = 0

            for i in range(0, max_iter):
                a = self.agent.choose_action(s, w_weighted, edge_index, edge_weight, back_depot)

                # obtain the reward and next state and some other information
                s_, r, done, info = self.env.step(a)
                back_depot = info[3]

                # Store the transition in memory
                self.agent.memory.push(s, a, r, s_, w_weighted, edge_index, edge_weight)
                self.agent.memory_counter += 1

                ep_r += r.item()

                # if the experience repaly buffer is filled, DQN begins to learn or update its parameters
                if self.agent.memory_counter > self.agent.mem_capacity:
                    self.agent.learn()
                    if done:
                        print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

                if done:
                    # if game is over, then skip the while loop.
                    print(" ->    Terminal event: episodic cumulative rewards = {}".format(ep_r))
                    break
                # use next state to update the current state.
                s = s_
            if self.render_on:
                self.env.render()
        return cumul_reward

    def loop(self, games, nbr_epoch, max_iter):
        cumul_reward_list = []

        # Start training
        print("\nCollecting experience...")
        for epoch_ in range(nbr_epoch):
            print(" -> epoch : " + str(epoch_))
            for g in range(1, games + 1):
                print(" -> games : " + str(g))
                cumul_reward = self.train(g, max_iter)
                cumul_reward_list.extend(cumul_reward)

                if self.verbose:
                    print(" <=> Finished game number: {} <=>\n".format(g))
