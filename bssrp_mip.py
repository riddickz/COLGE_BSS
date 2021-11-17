import numpy as np
import gurobipy as gp

from graph import Graph


class BSSRPMIP(object):

	def __init__(self, g, solver_time_limit=1e5, tol=1e-5):

		self.graph = g
		self.solver_time_limit = solver_time_limit
		self.tol = tol

		# collect needed info from instance 
		self.capacity = self.graph.max_load
		self.cost_matrix = self.graph.W_full
		self.demands= self.graph.demands
		self.num_vehicles = self.graph.num_vehicles
		self.time_limit = self.graph.time_limit

		# iterable lists
		self.V = list(range(self.graph.num_nodes))
		self.V_0 = list(range(1, self.graph.num_nodes))
		self.K = list(range(self.graph.num_vehicles))
		self.n_ports = self.graph.num_nodes - 1

		# TODO:
		self.tau = 0

		self.build_model()

	def optimize(self):
		self.model.optimize()



	def build_model(self):

		# initialize model
		self.model = gp.Model()
		self.model.setParam('TimeLimit', self.solver_time_limit)

		self.add_variables()
		self.add_depot_constraints()
		self.add_node_flow_constraints()
		self.add_demand_constraints()
		self.add_time_constraints()
		self.add_subtour_elimination_constraints()


	def add_variables(self):

		# initialize variables
		self.x_vars = {} # tour (edge) vars
		self.y_vars = {} # visit vars
		self.z_vars = {} # demand vars
		self.f_vars = {} # vehicle vars

		self.alpha_vars = None # unmet demand vars
		self.beta_vars = None # time limit vars

		# add transport variables to model [x_ijk]
		for k in self.K:
			for i in self.V:
				for j in self.V:
					if i == j:
						continue
					x_name = f"x_{i}_{j}_{k}"
					self.x_vars[x_name] = self.model.addVar(obj=self.cost_matrix[i,j], vtype=gp.GRB.BINARY, name=x_name)

		# add visiting variables to model [y_ik]
		for k in self.K:
			for i in self.V_0:
				y_name = f"y_{i}_{k}"
				self.y_vars[y_name] = self.model.addVar(obj=0.0, vtype=gp.GRB.BINARY, name=y_name)

		# add loading variables to model [z_ijk]
		for k in self.K:
			for i in self.V:
				for j in self.V:
					if i == j:
						continue
					z_name = f"z_{i}_{j}_{k}"
					self.z_vars[z_name] = self.model.addVar(obj=0.0, lb=0.0, vtype=gp.GRB.INTEGER, name=z_name)
		
		# add subtour elimination variables [f_ijk]
		for k in self.K:
			for i in self.V:
				for j in self.V:
					if i == j:
						continue
					f_name = f"f_{i}_{j}_{k}"
					self.f_vars[f_name] = self.model.addVar(obj=0.0, lb=0.0, vtype=gp.GRB.INTEGER, name=f_name)

		self.var_dict = {
			"x" : self.x_vars,
			"y" : self.y_vars,
			"z" : self.z_vars,
			"f" : self.f_vars
		}

		return 


	def add_depot_constraints(self):


		# constraint (2)
		for k in self.K:
			eq_ = 0
			for i in self.V_0:
				eq_ += self.x_vars[f"x_{0}_{i}_{k}"]
			self.model.addConstr(eq_ <= self.num_vehicles, name=f"2_depot_out_{k}")
			
		# constraint (3)
		for k in self.K:
			eq_ = 0
			for i in self.V_0:
				eq_ += self.x_vars[f"x_{i}_{0}_{k}"]
			self.model.addConstr(eq_ <= self.num_vehicles, name=f"3_depot_in_{k}")


	def add_node_flow_constraints(self):

		# constraint (4)
		for k in self.K:
			for i in self.V_0:
				eq_ = - self.y_vars[f"y_{i}_{k}"]
				
				for j in self.V:
					if i == j:
						continue    
					eq_ += self.x_vars[f"x_{i}_{j}_{k}"] 
				
				self.model.addConstr(eq_ == 0, name=f"4_node_out_{i}_{k}")

		# constraint (5)
		for k in self.K:
			for i in self.V_0:
				eq_ = - self.y_vars[ f"y_{i}_{k}"] 
				
				for j in self.V:
					if i == j:
						continue    
					eq_ += self.x_vars[f"x_{j}_{i}_{k}"] 
				
				self.model.addConstr(eq_ == 0, name=f"5_node_in_{i}_{k}")
			
		# constraint (6)
		for i in self.V_0:
			eq_ = 0
			for k in self.K:
				eq_ += self.y_vars[f"y_{i}_{k}"] 
			self.model.addConstr(eq_ == 1, name=f"6_node_visited_{i}")



	def add_demand_constraints(self):

		# constraint (8)
		for k in self.K:
			for i in self.V:
				for j in self.V:
					if i == j:
						continue
					x_name = f"x_{i}_{j}_{k}"
					z_name = f"z_{i}_{j}_{k}"
					self.model.addConstr(self.z_vars[z_name] - self.capacity*self.x_vars[x_name] <= 0,
						name=f"8_max_load_{i}_{j}_{k}")
					
		# constraint (9)
		for k in self.K:
			for i in self.V_0:
				eq_ = - self.demands[i] * self.y_vars[f"y_{i}_{k}"]
				for j in self.V:
					if i == j:
						continue
					eq_ += self.z_vars[f"z_{i}_{j}_{k}"]
				for h in self.V:
					if i == h:
						continue
					eq_ -= self.z_vars[f"z_{h}_{i}_{k}"]
				self.model.addConstr(eq_ == 0, name=f"9_bike_loading_{i}_{k}")

	def add_time_constraints(self):

		# constraint (10)
		for k in self.K:
			eq_ = 0
			for i in self.V:
				for j in self.V:
					if i == j:
						continue
					eq_ += self.cost_matrix[i,j] * self.x_vars[f"x_{i}_{j}_{k}"]
			for i in self.V_0:
				eq_ += self.tau * np.abs(self.demands[i]) * self.y_vars[f"y_{i}_{k}"]
				
			self.model.addConstr(eq_ <= self.time_limit, name=f"10_time_{k}")


	def add_subtour_elimination_constraints(self):

		# constraiant (11)
		for k in self.K:
			for j in self.V_0:
				self.model.addConstr(self.f_vars[f"f_0_{j}_{k}"] == 0, name=f"11_st_elim_zero_0_{j}_{k}")

				
		# constraiant (12)
		eq_ = 0
		for k in self.K:
			for j in self.V_0:
				eq_ += self.f_vars[f"f_{j}_0_{k}"]
		self.model.addConstr(eq_ == self.n_ports, name=f"12_st_elim_sum_f_eq_n")
					
		# constraiant (13)
		for k in self.K:
			for i in self.V:
				for j in self.V: 
					if i == j:
						continue
					f = self.f_vars[f"f_{i}_{j}_{k}"]
					x = self.x_vars[f"x_{i}_{j}_{k}"]
					self.model.addConstr(f - self.n_ports*x <= 0, name=f"13_st_elim_bound_{i}_{j}_{k}")
						
				
		# constraiant (15)
		for k in self.K:
			for i in self.V_0:
				eq_ = 0
				for j in self.V: 
					if i == j:
						continue
					eq_ += self.f_vars[f"f_{i}_{j}_{k}"]
					
				for h in self.V:
					if i == h:
						continue
					eq_ -= self.f_vars[f"f_{h}_{i}_{k}"]
					
				eq_ -= self.y_vars[ f"y_{i}_{k}"]
				self.model.addConstr(eq_ == 0, name=f"15_st_elim_one_diff_{i}_{k}")


	def get_next_node(self, edges, node):
	
		for edge in edges:
			if node == edge[0]:
				return edge[1], edge
			
		print(node, edges)
		raise Exception("Start node not found")

	def find_cycle(self, edges, start_node):
		
		cycle = [start_node]
		cur_node = start_node
		
		while True:
			
			next_node, cur_edge = self.get_next_node(edges, cur_node)
			edges.remove(cur_edge)
			cycle.append(next_node)
			
			if next_node == start_node:
				break
		
			cur_node = next_node
			
		return cycle

	def find_all_cycles(self, edges_):
		
		if len(edges_) == 0:
			return []
		
		edges = edges_.copy()
		cycles = []
		
		start_node = edges[0][0]
		cycle = self.find_cycle(edges, start_node)
		
		assert(len(edges) == 0)
		
		return cycle

	def get_routes(self):
		
		routes = {}

		for k in range(self.num_vehicles):
			edges = []
			for var in self.x_vars.values():
				if "x" in var.varName and np.abs(var.x - 1) < self.tol and k == int(var.varName.split('_')[-1]):
					i = int(var.varName.split("_")[1])
					j = int(var.varName.split("_")[2])
					edges.append((i,j))

			routes[k] = self.find_all_cycles(edges)
		
		self.routes = routes

		return routes

	def get_cost_of_route(self, route):
	    if len(route) == 0:
	        return 0
	    cost = 0
	    for i in range(len(route) - 1):
	        cost += self.cost_matrix[route[i], route[i+1]]
	    return cost

	def print_routes(self):
		for k in self.K:
		    print(f'Vehicle {k}:')
		    print(f'    Route:', self.routes[k])
		    print(f'    Cost:', self.get_cost_of_route(self.routes[k]), '\n')