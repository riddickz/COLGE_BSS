import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    window = 5 #int(max_episodes/20)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9])
    rolling_mean = pd.Series(reward_list).rolling(window).mean()

    std = pd.Series(reward_list).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(reward_list)),rolling_mean-std, rolling_mean+std, color='orange', alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Episode Length')
    plt.grid()

    x = np.arange(len(reward_list))
    ax2.scatter(x, reward_list, s=1.3)
    ax2.plot(reward_list,alpha=0.3, linewidth=0.8)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Episode Length')

    ax1.grid()
    ax2.grid()
    fig.tight_layout(pad=2)
    # plt.show()
    plt.pause(0.001)
    plt.close()
