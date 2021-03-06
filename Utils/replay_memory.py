# Adaptado desde https://github.com/whathelll/DeepRLBootCampLabs/tree/master/pytorch/utils
from collections import namedtuple
import random
import numpy as np

Transition = namedtuple("Transition", ["s", "a", "s_1", "r", "done"])

class ReplayMemory(object):
    """Experience replay buffer que muestrea de manera uniforme."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        result = []
        for i in range(self.__len__()):
            result.append(self.memory[i].__str__() + " \n")
        return "".join(result)

    def push(self, item):
        """Guarda las transiciones en el buffer considerando el tamaño maximo."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Devuelve 256 (batch_size) transiciones aleatorias guardadas en el buffer."""
        out = random.sample(self.memory, batch_size)
        batched = Transition(*zip(*out))
        s = np.array(list(batched.s))
        a = np.array(list(batched.a))
        # a = np.expand_dims(np.array(list(batched.a)), axis=1)
        s_1 = np.array(list(batched.s_1))
        r = np.expand_dims(np.array(list(batched.r)), axis=1)
        done = np.expand_dims(np.array(list(batched.done)), axis=1)
        return [s, a, s_1, r, done]

