import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from datetime import datetime
import argparse

def visualize_2D(nodes, W, nodes_weight=None): # Plot tour
    plt.figure(figsize=(20,15))
    colors = ['red'] # First node as depot
    for i in range(len(nodes)-1):
        colors.append('blue')

    edgeSet = set()
    for row in range(W.shape[0]):
        for column in range(W.shape[1]):
            if W.item(row,column) == 1 and (column,row) not in edgeSet: #get rid of repeat edge
                edgeSet.add((row,column))

    for edge in edgeSet:
        X = nodes[edge, 0]
        Y = nodes[edge, 1]
        plt.plot(X, Y,"g-",lw=0.2)

    xs, ys = nodes[:,0], nodes[:,1]
    plt.scatter(xs, ys,  color=colors) # Plot nodes
    if nodes_weight is None:
        for i, (x, y) in enumerate(zip(xs, ys)):
            plt.text(x, y, str(i), color="black", fontsize=30) # Plot bikes
    else:
        for i, (x, y) in enumerate(zip(xs, ys)):
            plt.text(x, y, str(nodes_weight[i]), color="black", fontsize=8) # Plot bikes

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_reward(reward_list):
    plt.ion()
    plt.figure(1)
    window = int(len(reward_list)/20)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9])
    rolling_mean = pd.Series(reward_list).rolling(window).mean()

    std = pd.Series(reward_list).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(reward_list)),rolling_mean-std, rolling_mean+std, color='orange', alpha=0.2)
    ax1.set_title('Episodic Reward Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Episodic Reward')
    plt.grid()

    x = np.arange(len(reward_list))
    ax2.scatter(x, reward_list, s=1.3)
    ax2.plot(reward_list,alpha=0.3, linewidth=0.8)
    ax2.set_title('Episodic Reward')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Episodic Reward')

    ax1.grid()
    ax2.grid()
    fig.tight_layout(pad=2)
    save_path = 'reward_{}.pdf'.format(timestamp())
    plt.savefig(save_path, bbox_inches='tight', dpi=200)

    plt.show()
    # plt.pause(0.001)
    plt.close()




def plot_loss(loss_list):
    plt.ion()
    plt.figure(1)
    window = int(len(loss_list) / 10)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9])
    rolling_mean = pd.Series(loss_list).rolling(window).mean()

    std = pd.Series(loss_list).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(loss_list)), rolling_mean - std, rolling_mean + std, color='orange', alpha=0.2)
    ax1.set_title('Loss Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Loss')
    plt.grid()

    x = np.arange(len(loss_list))
    ax2.scatter(x, loss_list, s=1.3)
    ax2.plot(loss_list, alpha=0.3, linewidth=0.8)
    ax2.set_title('Loss')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Loss')

    ax1.grid()
    ax2.grid()
    fig.tight_layout(pad=2)
    save_path = 'loss_{}.pdf'.format(timestamp())
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.show()
    # plt.pause(0.001)
    plt.close()

# Based on https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def timestamp():
    datetime.now(tz=None)
    timestampStr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # print('Current Timestamp : ', timestampStr)
    return timestampStr

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
