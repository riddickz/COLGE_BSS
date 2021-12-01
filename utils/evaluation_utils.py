import sys
sys.path.append('../')

import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import torch
import time

import runner
import agent

from graph import Graph
from baselines import BSSRPMIP
from baselines import NearestNeighboursHeuristic

from environment import Environment

tol = 1e-3

def demand_of_routes(routes, demands):
	""" Gets the demand of routes. """
	for i in range(len(routes)):
		route = routes[i]
		route_demand_order = list(map(lambda x: demands[x], route))
		print(f"Vehicle {i}:")
		print("    Route: ", route)
		print("    Demand:", route_demand_order)


def get_unvisited(routes, num_nodes):
	""" Gets the set of unvisited nodes, if any. """
	nodes = list(range(1, num_nodes))
	for route in routes:
		for node in route:
			if node == 0:
				continue
			else:
				nodes.remove(node)
				
	return nodes


def eval_mip_sol_in_env(mip, g):
	""" Evaluates MIP solution in environment. """
	graph_dict = {0 : g}
	env = Environment(graph_dict, "test", verbose=False)
	
	env.reset(0)
	done = False

	route_reward = {}
	for i, route in mip.routes.items():
		route_reward[i] = 0
		if len(route) == 0:
			continue

		for a in route[1:]:
			_, reward, done, _ = env.step(torch.tensor(a))
			route_reward[i] += reward
			if done:
				break
		if done:
			break
	
	unvisited_reward = 0
	while not done:
		_, reward, done, _ = env.step(torch.tensor(0))
		unvisited_reward += reward

	total_reward = unvisited_reward
	for i, reward in route_reward.items():
		total_reward += reward
	total_reward = total_reward.item() * env.reward_scale
		
		
	mip_obj = mip.model.objVal
	if np.abs(mip_obj + total_reward) > tol:
		print('ROUTE:', mip.routes)
		raise Exception(f"MIP objective and reward are not equal.\n  MIP objective: {mip_obj}\n  Reward: {total_reward}")
		
	specific_reward = {
		"tour" : env.ep_reward_tour,
		"demand" : env.ep_reward_demand,
		"overage" : env.ep_reward_overage,
	}
	
	return total_reward, env, specific_reward


def eval_nn_in_env(nn, g):
	""" Evaluates NN solution in environment. """
	graph_dict = {0 : g}
	env = Environment(graph_dict, "test", verbose=False)
	
	env.reset(0)
	done = False
	
	route_reward = {}
	for i, route in enumerate(nn.routes):
		route_reward[i] = 0
		if len(route) == 0:
			continue

		for a in route[1:]:
			_, reward, done, _ = env.step(torch.tensor(a))
			route_reward[i] += reward
			if done:
				break
		if done:
			break

	unvisited_reward = 0
	while not done:
		_, reward, done, _ = env.step(torch.tensor(0))
		unvisited_reward += reward
		
	total_reward = unvisited_reward
	for i, reward in route_reward.items():
		total_reward += reward

	total_reward = total_reward.item() * env.reward_scale
	
	specific_reward = {
		"tour" : env.ep_reward_tour,
		"demand" : env.ep_reward_demand,
		"overage" : env.ep_reward_overage,
	}
	
	return total_reward, env, specific_reward


def eval_agent_in_env(rl_agent, g, max_iters = 10000):
	""" Evaluates RL agent in environment. """
	graph_dict = {0 : g}
	env = Environment(graph_dict, "test", verbose=False)
	rl_runner = runner.Runner(env, rl_agent)
	reward, route = rl_runner.validate(0, max_iters, verbose=False, return_route=True)
	
	# reward
	reward = reward * env.reward_scale
	
	# get routes for each vehicle
	routes = []
	routes.append([0])
	route_num = 0
	for i in route[1:-1]:
		if i != 0:
			routes[route_num].append(i)
		else:
			routes[route_num].append(0)
			route_num += 1
			routes.append([0])
	routes[route_num].append(0)
	
	specific_reward = {
		"tour" : env.ep_reward_tour,
		"demand" : env.ep_reward_demand,
		"overage" : env.ep_reward_overage,
	}

	return reward, routes, env, specific_reward


def evaluate(g, n_instances, seed, rl_agent=None, mip_params=None, freq=10):
	""" Evaluates n_instances of each algorithms and stores results in dictionary. """
	g.seed(seed)
	results = {
		"demands" : [],
		"mip" : {"routes" : [], "cost" : [], "time" : [], "rewards" : [], },
		"nn" : {"routes" : [], "cost" : [], "time" : [], "rewards" : [], },
	}
	
	if rl_agent is not None:
		results["rl"] = {"routes" : [], "cost" : [], "time" : [], "rewards" : [],}
	
	for i in range(n_instances):
		if (i+1) % freq == 0:
			print(f"Instance: {i+1}/{n_instances}")
		
		g.bss_graph_gen()
		results["demands"].append(g.demands)
		
		# get MIP routes/reward
		mip = BSSRPMIP(g, **mip_params)
		mip_time = time.time()
		mip.optimize()
		mip_time = time.time() - mip_time
		mip_routes = mip.get_minimal_routes()
		mip_reward, _, mip_specific_reward = eval_mip_sol_in_env(mip, g)
		results["mip"]["routes"].append(mip_routes)
		results["mip"]["cost"].append(mip_reward)
		results["mip"]["time"].append(mip_time)
		results["mip"]["rewards"].append(mip_specific_reward)
		
		# get NN routes/reward
		nn = NearestNeighboursHeuristic(g, mip_params["visit_all"])
		nn_time = time.time()
		nn_routes = nn.run()
		nn_time = time.time() - nn_time
		nn_reward, _, nn_specific_reward = eval_nn_in_env(nn, g)
		results["nn"]["routes"].append(nn_routes)
		results["nn"]["cost"].append(nn_reward)
		results["nn"]["time"].append(nn_time)
		results["nn"]["rewards"].append(nn_specific_reward)
		
		# get RL routes/reward
		if rl_agent is not None:
			rl_time = time.time()
			rl_reward, rl_route, _, rl_specific_reward = eval_agent_in_env(rl_agent, g)
			rl_time = time.time() - rl_time            
			results["rl"]["routes"].append(rl_route)
			results["rl"]["cost"].append(rl_reward)
			results["rl"]["time"].append(rl_time)
			results["rl"]["rewards"].append(rl_specific_reward)

	return results
	

def render_mip(g, seed, mip_params=None, save_path=None):
	""" Renders a plot for the MIP. """
	g.seed(seed)
	g.bss_graph_gen()

	# get MIP routes/reward
	mip = BSSRPMIP(g, **mip_params)
	mip.optimize()
	mip_routes = mip.get_minimal_routes()
	mip_reward, mip_env, _ = eval_mip_sol_in_env(mip, g)

	mip_env.render(save_path=save_path)


	return


def render_nn(g, seed, mip_params=None, save_path=None):
	""" Renders a plot for the NN algorithm. """
	g.seed(seed)
	g.bss_graph_gen()

	# get NN routes/reward
	nn = NearestNeighboursHeuristic(g, mip_params["visit_all"])
	nn_routes = nn.run()
	nn_reward, nn_env, _ = eval_nn_in_env(nn, g)

	nn_env.render(save_path=save_path)

	return


def render_rl(g, seed, rl_agent, save_path=None):
	""" Renders a plot for the RL agent. """
	if rl_agent is None:
		return
	
	g.seed(seed)
	g.bss_graph_gen()

	rl_reward, rl_route, rl_env, _ = eval_agent_in_env(rl_agent, g)    

	rl_env.render(save_path=save_path)

	return


def print_results(results):
	""" Prints cost and time results. """
	name_map = {"rl" : "RL:  ", "nn" : "NN:  ", "mip" : "MIP: "}
	
	print("Reward:")
	for key, value in results.items():
		if key == "demands":
			continue
		print(f"  {name_map[key]} {np.mean(results[key]['cost'])} ")    
		
	print("Solving Time:")
	for key, value in results.items():
		if key == "demands":
			continue
		print(f"  {name_map[key]} {np.mean(results[key]['time'])} ")    
		

def get_reward_stats(results):
	""" Prints tables of breakdown of reward signals. """
	name_map = {"rl" : "RL", "nn" : "NN", "mip" : "MIP"}
	
	for key, value in results.items():
		if key == "demands":
			continue
			
		print(f"{name_map[key]} Rewards:")
		#print(value["rewards"])
		
		tours = list(map(lambda x: x["tour"], value["rewards"]))
		demands = list(map(lambda x: x["demand"], value["rewards"]))
		overage = list(map(lambda x: x["overage"], value["rewards"]))
		
		print(f"  Reward Type  | Mean               | std               ")
		print(f"  --------------------------------------------")
		print(f"  Tour:        | {np.mean(tours)}   | {np.std(tours)}          " )
		print(f"  Demands:     | {np.mean(demands)} | {np.std(demands)}" )
		print(f"  Overage:     | {np.mean(overage)} | {np.std(overage)} \n" )


def get_optimality_gaps(results, n_instances, rl_agent):
	""" Prints table of the optimality gaps to best known solutions. """
	best_known = []
	mip_gap = []
	nn_gap = []
	rl_gap = []
	
	for i in range(n_instances):
		mip_sol = results["mip"]["cost"][i]
		nn_sol = results["nn"]["cost"][i]
		
		if rl_agent is None:
			best_sol = max([mip_sol, nn_sol])
			
		else:
			rl_sol = results["rl"]["cost"][i]
			best_sol = max([mip_sol, nn_sol, rl_sol])
		
		best_known.append(best_sol)
		mip_gap.append(100*np.abs(best_sol - mip_sol) / np.abs(best_sol))
		nn_gap.append(100*np.abs(best_sol - nn_sol) / np.abs(best_sol))
		
		if rl_agent is not None:
			rl_gap.append(100*np.abs(best_sol - rl_sol) / np.abs(best_sol))
		
	
	print(f"  Method      | Gap  (%)            ")
	print(f"  ----------------------------------")
	print(f"  MIP:        | {np.mean(mip_gap)} ")
	print(f"  NN:         | {np.mean(nn_gap)} ")
	if rl_agent is not None:
		print(f"  RL:         | {np.mean(rl_gap)} ")


def plot_num_routes(results, rl_agent):
	""" Plots histogram of the number of vehicles used. """
	num_routes_mip = list(map(lambda x: len(x), results["mip"]["routes"]))
	num_routes_nn = list(map(lambda x: len(x), results["nn"]["routes"]))
	if rl_agent is not None:
		num_routes_rl = list(map(lambda x: len(x), results["rl"]["routes"]))
		fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True)
	else:
		fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True)
		
	axes[0].hist(num_routes_mip)
	axes[0].set_xlabel("Number of Vehicles")
	axes[0].set_ylabel("Frequency")
	axes[0].set_title("MIP")

	axes[1].hist(num_routes_nn)
	axes[1].set_xlabel("Number of Vehicles")
	axes[1].set_ylabel("Frequency")
	axes[1].set_title("NN Heuristic")

	if rl_agent is not None:

		axes[2].hist(num_routes_rl)
		axes[2].set_xlabel("Number of Vehicles")
		axes[2].set_ylabel("Frequency")
		axes[2].set_title("RL")

	fig.set_figheight(5)
	if rl_agent is not None:
		fig.set_figwidth(15)
	else:
		fig.set_figwidth(10)

	fig.suptitle("Histogram of the number of vehicles")
	plt.show()