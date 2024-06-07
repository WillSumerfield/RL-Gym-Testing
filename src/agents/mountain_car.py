"""
My attempt at solving a the Mountain Car problem with a Deep Q Network based on the theory introduced in this paper:
https://arxiv.org/pdf/1312.5602

DQN Summary:
A DQN is a 'Deep Quality Network'. That means:
 - Deep = It uses Neural Networks
 - Quality Network = We estimate the quality of each state/action pair to determine which actions to take.

DQN models only output discrete values,
and it is also "Off-Policy", which means that it updates the model based on expected future actions/rewards, but not necessarily the actions that the model ended up picking!
This is good because it allows our goal to remain "Find the optimal solution", but our agent gets to explore and take bad actions. This also means it won't avoid
actions nearby very bad actions, which can make our agent too 'cautious'. 

DQNs have two models - an actor and a target actor.
The actor is what plays the game - it explores the environment and tries to earn reward. 
The target actor is what we use to predict the future value. If we used the normal actor, we would be updating our evaluator as we trained which could make us over-value
action values (if we evaluate ourselves partially based off ourselves, we can make a positive feedback loop and over-value things)
Both models have the same inputs and outputs. They take in states, and output an expected reward for each action it can take.

The target actor is only used in the training process. It is meant to be a more stable, less quickly updated version of the actor. We only update them it a 
little bit based on changes we make to the actor, which is why they're stable - we don't change them too much too fast. So why don't we use these models for 
the actual acting? That's because we *like* a little instability in our actor and critic. We want them to explore and not get hung up on pre-conceived ideas 
about what action/states are good or bad. Also for the resasons states in the paragraph above.

We also use a "Relay Buffer", which means that we record every state, action, reward, and next state (called a transition), and then train the models at the end of each game. 
(You can also train models at the end of each step. I think this is more efficient data-wise, but you get less variation in your data too)
We also don't remove items from the buffer when we do this, so we re-use old samples. A big reason for using this is to break correlation beteween consecutive transitions. 
This mountain car game has a lot of really similar states which all happen in series (imagine how similar the states are when the car is slowly gaining speed to climb up 
the hill). This can create odd behaviors as the model over-attunes to those transitions. Keeping the transitions out of order can fix that issue. 

To go into more detail about how we use the target actor:
They are only used when taking our best guess at what the true value of an action/state pair is. We use this to update the actor. When we are figuring out how good the 
actor is, we first look at the reward we gained from the transition. Then we take our best guess at what the value of the next state would be, by using the target actor
to pick what action it would take in that situation. We compare that 'target Q value' to the Q value of the current actor to find the loss. We essentially assume that 
the target actor is perfect, and change the current actor based off of the loss between those values. The target actor isn't perfect of course, 
but they are approximations of the theoretically best Q values. If we do this enough, the model improves and should eventually converge on a good solution!
"""

import gymnasium as gym
import numpy as np
import os
import random
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from math import floor

from .agent import Agent


class MountainCar(Agent):

    ENVIRONMENT_NAME = "MountainCar-v0"

    SAVE_PATH = "/src/saves/mountain_car"
    ACTOR_MODEL_NAME = "actor"

    DEFAULT_EPISODES = 10000
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_HIDDEN_SIZES = [16,8]
    DEFAULT_EPSILON = 1
    DEFAULT_GAMMA = 0.9
    DEFAULT_TAU = 1e-2
    DEFAULT_LEARNING_RATE = 0.01


    """Estimates how familiar the model is with a point, given the model's history. It counts the number of times the agent has visited a nearby point. 
    Very similar to a Kernel Density Estimate."""
    class FamiliarityEstimation():

        def __init__(self, min: float, max: float, buckets: int):
            
            self.min = min
            self.max = max
            self.buckets = buckets
            self.total_points = 0

            # Create an even distribution of buckets along the statespace, one for each bucket. They all start with 0 visits.
            self.familiarity = [0 for bucket in range(buckets)]
    
        def add_point(self, point):
            bucket = self.nearest_bucket(point)
            self.familiarity[bucket] += 1
            self.total_points += 1

        def get_familiarity(self, point):
            bucket = self.nearest_bucket(point)
            if self.total_points == 0:
                return 0
            
            # The least familiar bucket returns 0, the most returns 1. Linear interpolation for buckets in between.
            sorted_familiarity = np.argsort(self.familiarity)
            for b in range(self.buckets):
                if bucket == sorted_familiarity[b]:
                    return b/self.buckets

        def nearest_bucket(self, point):
            return floor((point - self.min) / (self.max-self.min)*self.buckets)


    class DQN():

        # Takes in actions as inputs, outputs an acation. Used to choose which actions to take during a run.
        class Actor(nn.Module):
            
            def __init__(self, device, input_size, hidden_sizes, output_size, learning_rate):
                super(MountainCar.DQN.Actor, self).__init__()

                self.hidden_count = len(hidden_sizes)

                # Create the layers of the model
                self.input = nn.Linear(input_size, hidden_sizes[0]).to(device)
                self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]).to(device) for i in range(self.hidden_count-1)])
                self.output = nn.Linear(hidden_sizes[-1], output_size).to(device)

                # Set the optimizer
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

                # The loss function
                self.loss = nn.MSELoss()


            def forward(self, state) -> torch.Tensor:
                x = F.relu(self.input(state))
                for hidden_layer in self.hidden_layers:
                    x = F.relu(hidden_layer(x))
                x = self.output(x)

                return x
            

        """Copies the parameters from the src model to the dst model. The tau parameter determines the proportion of the model which is copied over.
        When tau=1, the dst model is set equal to the src model."""
        def copy_model(source_model: nn.Module, destination_model: nn.Module, tau=1.0) -> None:
            for source_param, dest_param in zip(source_model.parameters(), destination_model.parameters()):
                dest_param.data.copy_(tau*source_param.data + (1.0-tau)*dest_param.data)


        def __init__(self, device: torch.device, state_space: int, action_space: int,  hidden_sizes: list, epsilon: float, learning_rate: float, gamma: float, tau: float, train=True):
            
            self.device = device
            self.state_space = state_space
            self.action_space = action_space
            self.hidden_sizes = hidden_sizes
            self.epsilon = epsilon

            # Create a new actor
            if train:

                # How quickly the actor updates to match new data
                self.learning_rate = learning_rate

                # The discount factor on future rewards - changes how much we value expected future reward in the current state.
                self.gamma = gamma

                # The rate at which the target models are adjusted when we update the current models
                self.tau = tau

                # Randomly Initialize actor
                """Takes in states and outputs the assessed quality of each action. Also called an "action policy"."""
                self.actor = MountainCar.DQN.Actor(device, state_space, hidden_sizes, action_space, self.learning_rate).to(device)

                # Create target actor and target critic networks. In the beginning, they are copies of the actor and critic
                """The actor we train, but don't use for decision making. We update this whenever we update the normal actor, but just a little bit.
                This ensures that the target actor remains up to date, but isn't as 'jumpy' as the regular actor. This actions we get from this model are 
                what makes DDPG an off-policy model."""
                self.target_actor = MountainCar.DQN.Actor(device, state_space, hidden_sizes, action_space, self.learning_rate).to(device)
                MountainCar.DQN.copy_model(self.actor, self.target_actor)
        
        # Given a gamestate, find the highest estimated Q value.
        def take_action(self, env, state):

            # Choose the action we think is best
            if np.random.rand() > self.epsilon:
                with torch.no_grad():
                    return self.actor.forward(torch.tensor(state, device=self.device, dtype=torch.float)).argmax().item()
                
            # Get a random action
            else:
                return env.action_space.sample()
            

        # Update actor, and then update the target actor once the batch is done
        def update(self, memory_batch, update):

            # Batch the Q value estimates and targets
            q_values = []
            q_targets = []

            # Find the Q value estimates and targets for each memory in the batch
            for state, action, reward, next_state, win in memory_batch:

                state = torch.tensor(state, device=self.device, dtype=torch.float)
                next_state = torch.tensor(next_state, device=self.device, dtype=torch.float)
                
                # Find the expected future reward. If it won this transition, don't try to predict on the next state
                q_target = torch.tensor(reward, device=self.device, dtype=torch.float)
                if not win:
                    with torch.no_grad():
                        q_target += self.gamma * self.target_actor.forward(next_state).max()

                # Find the Q value estimates of our model
                q_values.append(self.actor.forward(state))

                q_prime = self.actor(state)
                q_prime[action] = q_target
                q_targets.append(q_prime)

            """Find the current actor's Q values for each action, and compare with the reward + expected reward estimated by the taget actor.
            The 'loss' in our case is the Mean Squared Error of the difference between these two Quality assessments."""
            actor_loss = self.actor.loss(torch.stack(q_values), torch.stack(q_targets))

            # Perform backpropigation on the policy loss to update the target actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update the target networks a little based on the current models
            if update:
                MountainCar.DQN.copy_model(self.actor, self.target_actor)


    def __init__(self, device: torch.device, train: bool, param_dict: dict, verbose: bool=True):

        self.param_dict = param_dict

        self.device = device

        self.verbose = verbose

        self.model_name = self.get_parameter("model_name", "")
        self.model_name = self.model_name + "_" if self.model_name else ""

        """The replay buffer holds memories of all past transitions. We sample this randomly to update the model, which helps avoid redundancy and bias
        introduced by sequential transitions which are super similar."""
        self.replay_buffer = deque(maxlen=10000)

        # The chance to choose a random action for exploration
        if train:
            self.epsilon = self.get_parameter("epsilon", MountainCar.DEFAULT_EPSILON)
        else:
            self.epsilon = 0

        # How many episodes between renders
        self.render_freq = self.get_parameter("render_freq", None)

        # The number and size of hidden layers in the model
        self.hidden_sizes = self.get_parameter("hidden_sizes", MountainCar.DEFAULT_HIDDEN_SIZES)

        # The number of games to train the model on
        self.episodes = self.get_parameter("episodes", MountainCar.DEFAULT_EPISODES)

        # The number of memories to train the model on each update.
        self.batch_size = self.get_parameter("batch_size", MountainCar.DEFAULT_BATCH_SIZE)

        # How quickly the actor updates to match new data
        self.learning_rate = self.get_parameter("learning_rate", MountainCar.DEFAULT_LEARNING_RATE)

        # The amount of weight places on future rewards
        self.gamma = self.get_parameter("gamma", MountainCar.DEFAULT_GAMMA)

        # The amount to update the target models
        self.tau = self.get_parameter("tau", MountainCar.DEFAULT_TAU)

        # How quickly the actor learns
        self.learning_rate = self.get_parameter("learning_rate", MountainCar)

        # The model our agent trains on
        self.env: gym.Env = gym.make(MountainCar.ENVIRONMENT_NAME, render_mode=None if train else "human", disable_env_checker=True)

        # The agent which interacts with and learns from the environment
        self.agent = MountainCar.DQN(self.device, self.env.observation_space.shape[0], self.env.action_space.n, self.hidden_sizes, self.epsilon, self.learning_rate, self.gamma, self.tau, train=train)

        # If the agent is in testing mode, load the saved model
        if not train:
            self.load_model()
            self.agent.actor.eval()


    # Create and train a DQN model from scratch. Returns a dataframe containing the reward attained at each episode.
    def train(self) -> pd.DataFrame:

        # Record the reward, so that we can graph the changes in reward over time.
        reward_df = pd.DataFrame(columns=["episode", "reward"])

        # Explore the environment until the agent can repeatedly find the solution
        do_exploration = True

        # Track the proportion of recent wins to know when to stop
        win_queue = deque([False for i in range(100)])
        recent_win_count = 0
        total_win_count = 0

        # Until the actor can reach the edge of the state space (and beat the game), use a familiarity estimate to reward to incentivize exploration
        self.familiarity_estimate = self.FamiliarityEstimation(self.env.observation_space.low[0], self.env.observation_space.high[0], 20)

        # Play the game up to certain number of episodes
        for episode in range(self.episodes):

            # Display the epsiode progress
            if self.verbose:
                print(f"Episode {episode+1}  :  Agent Skill: {recent_win_count}%  :  Total Wins: {total_win_count}      ", end='\r')

            # Only render episodes occasionally
            if self.render_freq:
                _render_mode = "human" if ((episode+1) % self.render_freq) == 0 else None
                self.env = gym.make(MountainCar.ENVIRONMENT_NAME, render_mode=_render_mode, disable_env_checker=True)

            # Reset the state of the environment so we can play it again
            state, info = self.env.reset()

            # Make decisions each time-step of the game
            game_over: bool = False
            step = 0
            total_reward = 0
            while not game_over:
                step += 1

                # Select an action deterministically, but with chance epsilon choose a random action instead
                action = self.agent.take_action(self.env, state)

                # Take the action in the environment, and observe the new state
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = np.array(next_state)

                # Don't count fake rewards from learning
                total_reward += reward

                # If the agent hasn't won yet, use the familiarity estimate to reward the agent for exploring new regions
                if do_exploration:

                    # Reward the agent for exploring unfamiliar spaces
                    reward += 0.5-self.familiarity_estimate.get_familiarity(next_state[0])

                    # Update the familiarity estimate
                    self.familiarity_estimate.add_point(next_state[0])

                # Use a manually crafted reward function to give gradual progress instead of all or nothing rewards
                else:
                    reward += max(2*next_state[0], 0) + abs(next_state[1]*10) # Place value on moving right and going fast

                # Store the 'transition (current state, current action, achieved reward, new state) in the Replay Buffer
                self.replay_buffer.append([state, action, reward, next_state, terminated])

                # Check if the game is over. Replace the normal truncation time with a longer one.
                game_over = terminated or total_reward <= -1000
                if terminated:
                    recent_win_count += 1
                    total_win_count += 1

                    # Check if we're done exploring
                    if do_exploration and recent_win_count > 10:
                        do_exploration = False
                        self.replay_buffer.clear() # Remove memories of old reward function
                        print(f"Finished Exploring the State Space at Episode {episode}           \n")
                    
                # If the game ended, record the total reward to plot later. Update the number of recent wins
                if game_over:
                    reward_df.loc[episode] = [episode, total_reward]

                    # Find the number of wins in the last 100 games
                    win_queue.append(terminated)
                    last_win = win_queue.popleft()
                    if last_win:
                        recent_win_count -= 1

                # Now the next state is the current state
                state = next_state

            # If the agent has won all of the recent episodes, we're done.
            if recent_win_count == 90 and total_reward > -150:
                print(f"Finished Training at Episode {episode}                       \n")
                break

            # Update the actor and critic using the a random sample from the replay buffer. The random-ness avoids bias and keeps it sample-efficient.
            if len(self.replay_buffer) > self.batch_size:
                self.agent.update(random.sample(self.replay_buffer, self.batch_size), (episode % 250) == 0)
            
            # Decrease the chance of doing random actions after each episode
            self.agent.epsilon = max(self.agent.epsilon - 1/self.episodes, 0)

        return reward_df


    # Load the final model from the previous training run, and dipslay it playing the environment
    def test(self) -> None:

        state, info = self.env.reset()

        # Run the model until the environment ends
        game_over: bool = False
        step = 0
        while not game_over:
            step += 1

            # Select an action. Only select the best one here, since we're showing off the model's capabilities
            action = self.agent.take_action(self.env, state)

            # Take the action in the environment, and observe the new state
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Now the next state is the current state
            state = next_state

            # If the game is over, exit.
            if terminated or truncated:
                return

        return
    

    def save_model(self):
        save_path = os.getcwd() + MountainCar.SAVE_PATH
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.agent.actor.state_dict(), save_path + '/' + self.model_name + MountainCar.ACTOR_MODEL_NAME + ".pt")


    def load_model(self):
        save_path = os.getcwd() + MountainCar.SAVE_PATH
        if not os.path.exists(save_path + '/' + self.model_name + MountainCar.ACTOR_MODEL_NAME + '.pt'):
            raise Exception("There are no saved models of the expected name. Run in training mode first with the same model-name parameter.")
        self.agent.actor = MountainCar.DQN.Actor(self.device, self.agent.state_space, self.agent.hidden_sizes, self.agent.action_space, self.learning_rate).to(self.device)
        self.agent.actor.load_state_dict(torch.load(save_path + '/' + self.model_name + MountainCar.ACTOR_MODEL_NAME + '.pt'))