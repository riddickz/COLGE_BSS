import argparse
import agent
import environment
import runner
import graph
import logging
import numpy as np
import sys

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='bss', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--graph_nbr', type=int, default='1000', help='number of differente graph to generate for the training sample')
parser.add_argument('--model', type=str, default='GATv2', help='model name')
parser.add_argument('--ngames', type=int, metavar='n', default='500', help='number of games to simulate')
parser.add_argument('--nepisode', type=int, metavar='n', default='1000', help='max number of episodes per game')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per episode')
parser.add_argument('--epoch', type=int, metavar='nepoch',default=25, help="number of epochs")
parser.add_argument('--lr',type=float, default=5e-4,help="learning rate")
parser.add_argument('--bs',type=int,default=32,help="minibatch size for training")
parser.add_argument('--n_step',type=int, default=3,help="n step in RL")
parser.add_argument('--n_node', type=int, metavar='node_numbers',default=20, help="number of node in generated graphs")
parser.add_argument('--knn', type=int, metavar='k_neighbor_node',default=10, help="number of node's KNN in generated graphs")
parser.add_argument('--coeff_demand',type=float, default=2.,help="obj coeff, penalty_cost_demand")
parser.add_argument('--coeff_time',type=float, default=5.,help="obj coeff, penalty_cost_time")
parser.add_argument('--car_speed',type=float, default=30.)
parser.add_argument('--time_limit',type=float, default=60.)
parser.add_argument('--n_car', type=int, metavar='car_nums', default=3, help='number of vehicles used in game')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')



def main():
    args = parser.parse_args()
    logging.info('Loading graph: nodes{}, ngames {}, graph_nbr {}, knn {} '.format(args.n_node, args.ngames, args.graph_nbr, args.knn))
    graph_dic = {}

    for graph_ in range(args.graph_nbr):
        seed = np.random.seed(120 + graph_)

        graph_dic[graph_] = graph.Graph(num_nodes=args.n_node,
                                        k_nn=args.knn,
                                        num_vehicles=args.n_car,
                                        penalty_cost_demand=args.coeff_demand,
                                        penalty_cost_time=args.coeff_time,
                                        speed=args.car_speed,
                                        time_limit=args.time_limit)

    logging.info('Loading agent...')
    agent_class = agent.Agent(args.model, args.lr)

    logging.info('Loading environment %s' % args.environment_name)
    env_class = environment.Environment(graph_dic,args.environment_name)

    print("Running a single instance simulation...")
    my_runner = runner.Runner(env_class, agent_class, args.verbose, render = False)
    final_reward = my_runner.train_loop(args.ngames,args.epoch,args.nepisode, args.niter)
    print("Obtained a final reward of {}".format(final_reward))
    agent_class.save_model()



if __name__ == "__main__":
    main()
