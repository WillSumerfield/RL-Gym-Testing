""" A base class for all agents, which attempt to solve their environment."""

class Agent:

    def train(self):
        """Train in the environment and create a best agent."""
        raise NotImplementedError
    
    def test(self):
        """Creates the best agent and runs it in the environment."""
        raise NotImplementedError
    