import random


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self._capacity = capacity
        self._buffer = []
        self._position = 0

    def add(self, state, action, reward, next_state, done=0):
        if len(self._buffer) < self._capacity:
            self._buffer.append((state, action, reward, next_state, done))
        else:
            self._buffer[self._position] = (state, action, reward, next_state, done)
        self._position = (self._position + 1) % self._capacity

    def sample(self, sample_size):
        samples = random.sample(self._buffer, sample_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for (state, action, reward, next_state, done) in samples:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        #print("BUFFER RETURNS: ")
        #print("states: ", states)
        #print("next_states: ", next_states)
        return states, actions, rewards, next_states, dones

    def get_size(self):
        return len(self._buffer)

    def get_capacity(self):
        return self._capacity
