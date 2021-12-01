# Reinforcement Learning for the Bike Sharing Systems

Bike sharing systems (BSS) have become a popular mode of public transport as they allow commuters to seamlessly borrow a bike from a docking station close to their departure and return it near their destination.  For service providers, a key aspect to maintain customer satisfaction is the ability to maintain properly balanced stations in order to ensure customers are able to pick up or drop off a bike at any station.  One approach to balance stations is through deployment of a set of vehicles that move bikes from stations with excess to those in need, which results in an optimization problem similar to the vehicle routing problem, where the objective depends on both reallocation and tour minimization.  In this work, we propose the use of reinforcement learning and graph neural networks to develop a heuristic for the BSS re-balancing problem. 

## Summary

The objective of this repo is to implement a reinforment learning agent for the bike sharing system routing problem, which involves complex routing decsions in conjunction with other reward signals such as time limits and meeting both positive and negative demands.  This problem can be viewed as a Markov Decision Process (MDP) and represented by the below image.  

![alt text](https://github.com/riddickzhou/COLGE_BSSVRP/blob/RL_bike_tmp/.idea/mdp.png)



## Structure of the code


### main.py

`main.py` allows to define arguments and launch the code.

### graph.py

Define the graph object, espacially the kind of degree distribution and the methods.

### runner.py 

This script calls each step of reinforcement learning part in a loop (epochs + games):
  - `observe`: get states features of the problem (`environment class`)
  - `act`: take an action from the last observation (`agent class`)
  - get `reward` and `done` information from the last action (`environment class`)
  - perform a learning step with last observation, last action, observation and reward  (`agent class`)
  
### agent.py

Define the agent object and methods needed in deep Q-learning algorithm.

### model.py

Define the Q-function and the embedding algorithm.

### environment.py

Define the environment object for the BSSrp.

### baselines/

Folder containing the exact MILP formulation of the BSSrp and a nearest neighbour heuristic.

### notebooks/

Folder containing the evaluation scripts.

## Acknowledgements

- The repo in which we built our implementation on https://github.com/louisv123/COLGE
- The repo in which COLGE is based on https://github.com/Hanjun-Dai/graph_comb_opt







  


