import matplotlib.pyplot as plt

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