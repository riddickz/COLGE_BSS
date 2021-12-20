"""
This is the machinnery that runs your agent in an environment.

"""
import numpy as np
import torch
from utils.vis import plot_reward, plot_loss,timestamp
import pickle
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class Runner:
    def __init__(self, environment, agent, verbose=False, render=False):
        self.env = environment
        self.agent = agent
        self.verbose = verbose
        self.render_on = render
        self.plot_on = False
        self.step_cnt = 0
        self.q_a = []

    def train(self, g, max_episode, max_iter, iter_count, writer):
        self.agent.policy_net.train() # dropout/BN train mode
        self.agent.target_net.train() # dropout/BN train mode

        reward_list = []
        loss_list = []
        epsilon_list = []

        for i_episode in range(max_episode):
            s, adj_mat,mask = self.env.reset(g)
            ep_r = 0
            ep_loss = []
            ep_eps = []


            for i in range(0, max_iter):
                mask = mask.to(device)
                a, _ = self.agent.choose_action(s, adj_mat, mask)
                #a, q_a = self.agent.choose_action(s, adj_mat, mask)
				
                # obtain the reward and next state and some other information
                s_, r, done, info = self.env.step(a)
                mask_ = info[3]
                self.step_cnt +=1

                if (not q_a is None) and self.step_cnt%100==0: # collecting q value info
                    self.q_a.append(q_a)
                    writer.add_histogram("q_a", q_a, self.step_cnt)

                # # Store the transition in memory
                # self.agent.memory.push(s, a, r, s_, adj_mat, mask)
                self.agent.replay_buffer.add(s, a, r, s_, adj_mat)
                self.agent.memory_counter += 1

                ep_r += r.item()

                # if the experience replay buffer is filled, DQN begins to learn or update its parameters
                if self.agent.memory_counter > self.agent.mem_capacity:
                    loss, epsilon =self.agent.learn(iter_count)
                    ep_loss.append(loss.item())
                    ep_eps.append(epsilon)

                    if done:
                        print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

                if done:
                    # if game is over, then skip the while loop.
                    print(" ->    Terminal event: episodic rewards = {}".format(ep_r))
                    break

                # use next state/mask to update the current state/mask.
                s = s_
                mask = mask_

            reward_list.append(ep_r*500)
            loss_avg = np.mean(ep_loss)
            eps_avg = np.mean(ep_eps)
            if len(ep_loss) != 0:
                loss_list.append(loss_avg)
                epsilon_list.append(eps_avg)

            # collect training tracking info
            writer.add_scalar("ep_r", ep_r*500, iter_count)
            writer.add_scalar('loss_avg', loss_avg, iter_count)
            writer.add_scalar('eps_avg', eps_avg, iter_count)

            for name, weight in self.agent.policy_net.named_parameters():
                if (weight.grad is not None) and iter_count % 100 == 0 \
                        and (weight.requires_grad) and ("bias" not in name):
                        writer.add_histogram(name, weight, iter_count)
                        writer.add_histogram(f'{name}.grad', weight.grad, iter_count)
            iter_count+=1

            # render trips
            if self.render_on:
                self.env.render()
                print("Rendering Env")

        return reward_list, loss_list, epsilon_list, iter_count

    def train_loop(self, games, max_epoch, max_episode=30, max_iter=1000):
        writer = SummaryWriter()
        cumul_reward_list = []
        cumul_loss_list = []
        cumul_epsilon_list = []
        CHECK =1000
        itr_count = 0 # episode counter for tensorboard
        # Start training
        print("\nCollecting experience...")
        for epoch_ in range(max_epoch):
            print(" -> epoch : " + str(epoch_))
            for g in range(games):
                print(" -> games : " + str(g))
                reward_list, loss_list, epsilon_list, itr_count = self.train(g, max_episode, max_iter, itr_count, writer)
                cumul_reward_list.extend(reward_list)
                cumul_loss_list.extend(loss_list)
                cumul_epsilon_list.extend(epsilon_list)
                self.agent.scheduler.step()


                if self.plot_on:
                    plot_reward(cumul_reward_list)
                    plot_loss(cumul_loss_list[50:])
                    plot_loss(cumul_loss_list)

                if g == CHECK:
                    plot_reward(cumul_reward_list)
                    plot_loss(cumul_loss_list[50:])

                if self.verbose:
                    print(" <=> Finished game number: {} <=>\n".format(g))

        pickle.dump(cumul_reward_list, open('rl_results/reward_{}.pkl'.format(timestamp()), 'wb'))
        pickle.dump(cumul_loss_list, open('rl_results/loss_{}.pkl'.format(timestamp()), 'wb'))
        #pickle.dump(self.q_a, open('rl_results/q_a{}.pkl'.format(timestamp()), 'wb'))


        plot_reward(cumul_reward_list)
        plot_loss(cumul_loss_list)
        self.env.render()
        writer.close()
        return cumul_reward_list, cumul_loss_list, cumul_epsilon_list

    def validate(self, g, max_iter, verbose=True, return_route=False):
        self.agent.policy_net.eval() # dropout/BN eval mode
        self.agent.target_net.eval() # dropout/BN eval mode

        s, adj_mat,mask = self.env.reset(g)
        ep_r = 0
        route = [0]

        for i in range(0, max_iter):
            a, _ = self.agent.choose_action(s, adj_mat, mask)
            route.append(a.item())
            s_, r, done, info = self.env.step(a)

            ep_r += r.item()

            if done:
                if verbose:
                    print(" ->    Terminal event: episodic rewards = {},tour= {}, dem = {}, overage ={}".format(ep_r,self.env.ep_reward_tour,self.env.ep_reward_demand,self.env.ep_reward_overage))
                break

            s = s_
            mask = info[3]

        route.append(0)

        if self.render_on:
            self.env.render()

        if return_route:
            return ep_r, route

        return ep_r

    def validate_loop(self, games, max_iter=1000):
        self.agent.epsilon_ = 0
        reward_list = []
        for g in range(games):
            print(" -> games : " + str(g))
            ep_r = self.validate(g, max_iter)
            reward_list.append(ep_r)

            with open('val_result.pickle', 'wb') as handle:
                pickle.dump(reward_list, handle)

        return reward_list