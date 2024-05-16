""" A base class for all agents, which attempt to solve their environment."""

import pandas as pd


class Agent:


    def train(self) -> pd.DataFrame:
        """Train in the environment and create a best agent.
        Returns the DataFrame of the reward over episodes."""
        raise NotImplementedError
    
    def test(self):
        """Creates the best agent and runs it in the environment."""
        raise NotImplementedError
    
    def save_model(self):
        """Creates the best agent and runs it in the environment."""
        raise NotImplementedError
    
    def load_model(self):
        """Creates the best agent and runs it in the environment."""
        raise NotImplementedError

    # A helper function used to retrieve values from the parameter dict
    def get_parameter(self, parameter_name: str, default: any):
        val = self.param_dict[parameter_name]
        if val is None:
            val = default
        return val