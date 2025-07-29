import random
from collections import deque

class replayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add a new experience to the buffer."""
        self.buffer.append(experience)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from the buffer."""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in the buffer to sample a batch.")
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)