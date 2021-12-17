import random
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'adj', 'mask'))

# replay option 1 without PER
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_list(self, size):
        x = []
        for _ in range(size):
            i = random.randrange(len(self.memory) - 5)
            x.append(self.memory[i])
        return x

    def __len__(self):
        return len(self.memory)

# replay option 2 with PER
class ReplayBuffer:
    def __init__(self, capacity, alpha, num_node, num_ft):
        # We use a power of 2 for capacity because it simplifies the code and debugging
        self.capacity = capacity
        self.alpha = alpha
        self.num_node = num_node
        self.num_ft = num_ft

        # Maintain segment binary trees to take sum and find minimum over a range
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        # Current max priority, p, to be assigned to new transitions
        self.max_priority = 1.

        self.data = {
            'obs': np.zeros(shape=(capacity, self.num_ft, self.num_node), dtype=np.float32),
            'action': np.zeros(shape=capacity, dtype=np.int64),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, self.num_ft, self.num_node), dtype=np.float32),
            'adj': np.zeros(shape=(capacity, self.num_node, self.num_node), dtype=np.float32),
        }

        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def add(self, obs, action, reward, next_obs, adj):
        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['adj'][idx] = adj

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity

        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        # pαi, new samples get `max_priority`
        priority_alpha = self.max_priority ** self.alpha

        # Update the two segment trees for sum and minimum
        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):

        # Leaf of the binary tree
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        # Update tree, by traversing along ancestors. Continue until the root
        while idx >= 2:
            idx //= 2  # Get the index of the parent node

            # Value of the parent node is the minimum of it's two children
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.priority_sum[idx] = priority

        # Update tree, by traversing along ancestors. Continue until the root
        while idx >= 2:
            idx //= 2  # Get the index of the parent node
            # Value of the parent node is the sum of it's two children
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        # The root node keeps the sum of all values
        return self.priority_sum[1]

    def _min(self):
        # The root node keeps the minimum of all values
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1  # Start from the root
        while idx < self.capacity:
            # If the sum of the left branch is higher than required sum
            if self.priority_sum[idx * 2] > prefix_sum:
                # Go to left branch of the tree
                idx = 2 * idx
            else:
                # Otherwise go to right branch and reduce the sum of left branch from required sum
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        # We are at the leaf node. Subtract the capacity by the index in the tree to get the index of actual value
        return idx - self.capacity

    def sample(self, batch_size, beta):
        # Initialize samples
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        # Get sample indexes
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        prob_min = self._min() / self._sum()

        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            samples['weights'][i] = weight / max_weight

        # Get samples data
        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            # Set current max priority
            self.max_priority = max(self.max_priority, priority)

            # Calculate pαi
            priority_alpha = priority ** self.alpha
            # Update the trees
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        return self.capacity == self.size
