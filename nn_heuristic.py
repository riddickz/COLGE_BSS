import numpy as np
import gurobipy as gp

from graph import Graph


class NearestNeighboursHeuristic(object):
	
	def __init__(self, g, visit_all):
		""" Nearest Neighbours Heurisitic for BSSrp. """
		self.graph = g
		self.visit_all = visit_all
		
		self.num_nodes = self.graph.num_nodes
		self.capacity = self.graph.max_load
		self.cost_matrix = self.graph.W_full
		self.demands = self.graph.demands
		self.num_vehicles = self.graph.num_vehicles
		self.time_limit = self.graph.time_limit

		self.nodes_visited = []
		self.nodes_to_visit = set(range(1, self.num_nodes))
		

	def run(self):
		""" Runs the nearest neighbours heuristic. """
		self.routes = self.get_routes()
		return self.routes

	def get_routes(self):
		""" Gets all routes. """
		routes = []
		if not self.visit_all:
			for i in range(self.num_vehicles):
				if len(self.nodes_to_visit) == 0:
					break
				routes.append(self.get_single_route())	
		else:
			while True:
				if len(self.nodes_to_visit) == 0:
					break
				routes.append(self.get_single_route())
		return routes
		
	def get_single_route(self):
		""" Gets a single route. """
		
		load = self.graph.num_start
		route = [0]
		route_time = 0
		node = 0
		
		while True:
			
			# visited all nodes or time limit reached
			if len(self.nodes_to_visit) == 0 or self.is_time_limit(node, route_time):
				break
								
			# otherwise get next node
			next_node = self.get_next_node(node, load)

			if not self.visit_all and next_node == 0:
				break
			
			# update load
			if self.demands[next_node] > 0:
				load = min(load + self.demands[next_node], self.capacity)
			else:
				load = max(0, load + self.demands[next_node])
				
			# update route
			route_time += self.cost_matrix[node, next_node]
			route.append(next_node)
			self.nodes_to_visit.remove(next_node)
			node = next_node
			
   
		route.append(0)
		return route
			
			
	def is_time_limit(self, node, route_time):
		""" Chooses the next node based on proximity and load.  """
		if route_time + self.cost_matrix[node, 0] > self.time_limit:
			return True
		return False
			
		
	def get_next_node(self, node, load):
		""" Chooses the next node based on proximity and load.  """
		distances = {}
		demands = {}
		
		for next_node in self.nodes_to_visit:
			distances[next_node] = self.cost_matrix[node, next_node]
			demands[next_node] = self.demands[next_node]
			
		sorted_distances = sorted(distances.items(), key=lambda x: x[1])
			
		# find nearest node which capacity can be met
		candidates = []
		for next_node, demand in demands.items():
			
			# visitable without going over
			if demand > 0 and (self.capacity - load) > demand:
				candidates.append(next_node)
				
			# visitable without not meeting demand
			if demand < 0 and load > - demand:
				candidates.append(next_node)
				
	
		# go to closest node in set of candidates
		if len(candidates) > 0:
			for next_node, dist in sorted_distances:
				if next_node in candidates:
					return next_node

		# otherwise, just go to the nearest neighbour
		if self.visit_all:
			return sorted_distances[0][0]

		return 0
				
		
		