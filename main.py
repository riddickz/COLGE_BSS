import argparse
import agent
import environment
import runner
import graph
import logging
import numpy as np
import networkx as nx
import sys

# # 2to3 compatibility
# try:
#     input = raw_input
# except NameError:
#     pass

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='bss', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--graph_nbr', type=int, default='1000', help='number of differente graph to generate for the training sample')
parser.add_argument('--model', type=str, default='GCN_QN_1', help='model name') #GCN_QN_1 RecurrentGCN
parser.add_argument('--ngames', type=int, metavar='n', default='500', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
parser.add_argument('--epoch', type=int, metavar='nepoch',default=25, help="number of epochs")
parser.add_argument('--lr',type=float, default=5e-4,help="learning rate")
parser.add_argument('--bs',type=int,default=32,help="minibatch size for training")
parser.add_argument('--n_step',type=int, default=3,help="n step in RL")
parser.add_argument('--node', type=int, metavar='nnode',default=30, help="number of node in generated graphs")
parser.add_argument('--knn', type=int, metavar='kknn',default=10, help="number of node's KNN in generated graphs")
parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')

def main():
    args = parser.parse_args()
    logging.info('Loading graph: nodes{}, ngames {}, graph_nbr {} '.format(args.node, args.ngames, args.graph_nbr))
    graph_dic = {}

    for graph_ in range(args.graph_nbr):
        seed = np.random.seed(120 + graph_)
        graph_dic[graph_] = graph.Graph(num_nodes=20,
                                        k_nn=10,
                                        num_vehicles=3,
                                        penalty_cost_demand=1,
                                        penalty_cost_time=1,
                                        speed=30,
                                        time_limit=60)

    logging.info('Loading agent...')
    agent_class = agent.Agent(args.model, args.lr)

    logging.info('Loading environment %s' % args.environment_name)
    env_class = environment.Environment(graph_dic,args.environment_name)

    print("Running a single instance simulation...")
    my_runner = runner.Runner(env_class, agent_class, args.verbose)
    final_reward = my_runner.loop(args.ngames,args.epoch, args.niter)
    print("Obtained a final reward of {}".format(final_reward))
    agent_class.save_model()



if __name__ == "__main__":
    main()
