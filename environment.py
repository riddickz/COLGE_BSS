import numpy as np
import torch
import pulp

"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self, graph,name):
        self.graphs = graph
        self.name= name

    def reset(self, g):
        self.games = g
        self.graph_init = self.graphs[self.games]
        self.nodes = self.graph_init.nodes()
        self.nbr_of_nodes = 0
        self.edge_add_old = 0
        self.last_reward = 0
        self.observation = torch.zeros(1,self.nodes,1,dtype=torch.float)
        self.static = self.graph_init.static # location coordinates
        self.dynamic = self.graph_init.dynamic.int() # load, demand
        self.demand_init_tot = torch.abs(self.dynamic[1]).sum()
        self.max_load = self.graphs[self.games].max_load
        self.node_idx = 0


    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        return self.observation

    def act(self, chosen_idx):

        # # Update the dynamic elements differently for if we visit depot vs. a city
        # chosen_idx = torch.tensor(chosen_idx)
        # visit = chosen_idx.ne(0)
        # depot = chosen_idx.eq(0)

        # Clone the dynamic variable so we don't mess up graph
        all_loads = self.dynamic[0]
        all_demands = self.dynamic[1]
        load = all_loads[chosen_idx]
        demand = all_demands[chosen_idx]

        # Loading and supplying constraint
        # if we've chosen to visit a city, try to satisfy as much demand as possible
        if chosen_idx != 0:
            new_load = torch.clamp(load - demand, max= self.max_load, min=0)
            if demand >= 0:
                new_demand = torch.clamp(demand - load, min=0)
            else:
                new_demand = torch.clamp(demand + (self.max_load - load), max = 0)

            all_loads = new_load
            all_demands[chosen_idx] = new_demand


        # Return to depot to half fill vehicle load
        else:
            all_loads= int(self.max_load/2) # TODO: vehicle is reload scheme
            all_demands[chosen_idx] = 0 # Demand at depot is 0

        # print("all_demands:" ,all_demands)
        # print("all_loads:" ,all_loads)

        self.observation[:, chosen_idx, :] = 1
        self.dynamic[0] = all_loads
        self.dynamic[1] = all_demands

        reward = self.get_reward(chosen_idx)
        self.node_idx = chosen_idx

        return reward

    def get_reward(self, node):

        if self.name == "MVC":

            new_nbr_nodes=np.sum(self.observation[0].numpy())

            if new_nbr_nodes - self.nbr_of_nodes > 0:
                reward = -1#np.round(-1.0/20.0,3)
            else:
                reward = 0

            self.nbr_of_nodes=new_nbr_nodes

            #Minimum vertex set:

            done = True

            edge_add = 0

            for edge in self.graph_init.edges():
                if self.observation[:,edge[0],:]==0 and self.observation[:,edge[1],:]==0:
                    done=False
                    # break
                else:
                    edge_add += 1

            #reward = ((edge_add - self.edge_add_old) / np.max(
            #   [1, self.graph_init.average_neighbor_degree([node])[node]]) - 10)/100

            self.edge_add_old = edge_add

            return (reward,done)

        elif self.name=="MAXCUT" :

            reward=0
            done=False

            adj= self.graph_init.edges()
            select_node=np.where(self.observation[0, :, 0].numpy() == 1)[0]
            for nodes in adj:
                if ((nodes[0] in select_node) & (nodes[1] not in select_node)) | ((nodes[0] not in select_node) & (nodes[1] in select_node))  :
                    reward += 1#/20.0
            change_reward = reward-self.last_reward
            if change_reward<=0:
                done=True

            self.last_reward = reward

            return (change_reward,done)

        elif self.name == "bss":
            done=False
            loc_prev = self.static[self.node_idx]
            loc = self.static[node]
            dist = (loc_prev-loc).norm()

            demand = np.abs(self.dynamic[1]).sum()
            reward = -dist # TODO add soft coeff

            if demand == 0 or self.observation.sum() == self.observation.shape[1]:
            # if demand == 0:
                demand_penality = demand#/self.demand_init_tot # #TODO better normalization
                reward += demand_penality #TODO better unsat final demand penality
                done =True
                print("Games Finished: ", self.games )
            # print("visit node:{}, reward:{}, done:{}:".format(node, reward, done))
            # print("$"*50)

            return (reward, done)


    def get_approx(self):

        if self.name=="MVC":
            cover_edge=[]
            edges= list(self.graph_init.edges())
            while len(edges)>0:
                edge = edges[np.random.choice(len(edges))]
                cover_edge.append(edge[0])
                cover_edge.append(edge[1])
                to_remove=[]
                for edge_ in edges:
                    if edge_[0]==edge[0] or edge_[0]==edge[1]:
                        to_remove.append(edge_)
                    else:
                        if edge_[1]==edge[1] or edge_[1]==edge[0]:
                            to_remove.append(edge_)
                for i in to_remove:
                    edges.remove(i)
            return len(cover_edge)

        elif self.name=="MAXCUT":
            return 1

        else:
            return 'you pass a wrong environment name'

    def get_optimal_sol(self):

        if self.name =="MVC":

            x = list(range(self.graph_init.g.number_of_nodes()))
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMinimize)

            mdl += sum(xv[k] for k in xv)

            for edge in self.graph_init.edges():
                mdl += xv[edge[0]] + xv[edge[1]] >= 1, "constraint :" + str(edge)
            mdl.solve()

            #print("Status:", pulp.LpStatus[mdl.status])
            optimal=0
            for x in xv:
                optimal += xv[x].value()
                #print(xv[x].value())
            return optimal

        elif self.name=="MAXCUT":

            x = list(range(self.graph_init.g.number_of_nodes()))
            e = list(self.graph_init.edges())
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)
            ev = pulp.LpVariable.dicts('ev', e,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMaximize)

            mdl += sum(ev[k] for k in ev)

            for i in e:
                mdl+= ev[i] <= xv[i[0]]+xv[i[1]]

            for i in e:
                mdl+= ev[i]<= 2 -(xv[i[0]]+xv[i[1]])

            #pulp.LpSolverDefault.msg = 1
            mdl.solve()

            # print("Status:", pulp.LpStatus[mdl.status])

            return mdl.objective.value()


