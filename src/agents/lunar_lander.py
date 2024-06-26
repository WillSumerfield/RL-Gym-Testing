"""
My attempt at solving a the Lunar Lander problem with a Proximal Policy Optimization model based on the theory introduced in this paper:
https://arxiv.org/abs/1707.06347

PPO Summary:
A PPO is 'Proximal Policy Optimization'. That means:
 - Proximal = We don't optimize with respect to the raw reward, instead we optimize a "surrogate objective".
 - Policy Optimization = We optimize the policy pi based on rewards, as opposed to optimizing a value function like in Q-Learning.

These models can not only deal with continuous action spaces, but have higher reliability, meaning that they don't get 'stuck' with bad policies as often.
It's really good at utilizing data, making it a good fit for applications where data is expensive (like robotics). That attribute is not as applicable in these simulations.
This model is stochastic, meaning that actions are chosen by converting the output of the actor to a probability distribtion, and then picking one randomly.
It is also "On-Policy", which means that it updates the model based on actual decisions the model makes. 
It is also "Online", meaning that it learns as it plays, rather than learning after a long session of playtime to begin training.

PPOs have two models - an actor and a critic.
The actor is what plays the game - it explores the environment and tries to earn reward. 
We use the critic to evaluate the actor - this is what lets us know how to improve the actor. The critic tries to estimate the actor's average received value of each state,
based the reward recieved (and the expected value from future states). 
However, what the actor learns isn't a simple reward function. Instead, the actor learns a more complex formula called the surrogate objective (surrogate because it isn't just TD learning or straight reward).
We use the critic's value guesses about state-action pairs the to tell the actor what it does well, and what it should do differently. 

The PPO works by playing a small number of steps (the batch size), and then training multiple times on that batch of data. This is why the PPO is so data efficient; it re-uses data.
The reason that it can re-use data better than other models can is because it scales the amount it learns based on how different the current model is from when the data is collected.
The model is only sort of "Online" - it doesn't train after each action, instead it trains after a relatively small number of actions, called a batch. This is why it is important to
check how different the model is after each training step - each time we train on an item from the batch, the policy gets more and more different from the policy used to make those
actions, and the data becomes less relevant.
PPOs also limit how much it will increase the likelyhood of an action, so avoid overfitting on high rewards on single steps. 

The surrogate objective is: min( r*A, clip(r, 1-e, 1+e)*A )
Where r is the difference between the policy used to , e is some small number, and A is the advantage for the current state-action pair. A = discounted rewards - critic's estimate
The "clip" is what limits change based on model difference, and the "min" ensures that we don't increase the likelyhood of any action too much in a single step.
"""

import gymnasium as gym
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical

from .agent import Agent


class LunarLander(Agent):

    ENVIRONMENT_NAME = "LunarLander-v2"

    SAVE_PATH = "/src/saves/lunarlander"
    ACTOR_MODEL_NAME = "actor"
    CRITIC_MODEL_NAME = "critic"

    DEFAULT_EPISODES = 3500
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_EPOCHS = 3
    DEFAULT_HIDDEN_SIZES = [[64,32,16], [64,32,16]]
    DEFAULT_EPSILON = 0.2
    DEFAULT_LEARNING_RATE = [1e-4, 1e-3]
    DEFAULT_GAMMA = 0.99


    class PPO():

        # Takes in actions as inputs, outputs an acation. Used to choose which actions to take during a run.
        class Actor(nn.Module):
            
            def __init__(self, device, input_size, hidden_sizes, output_size, learning_rate):
                super(LunarLander.PPO.Actor, self).__init__()

                self.hidden_count = len(hidden_sizes)

                # Create the model
                self.input = nn.Linear(input_size, hidden_sizes[0]).to(device)
                self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]).to(device) for i in range(self.hidden_count-1)])
                self.output = nn.Linear(hidden_sizes[-1], output_size).to(device)

                # Set the optimizer
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

            def forward(self, state) -> torch.Tensor:
                x = F.relu(self.input(state))
                for hidden_layer in self.hidden_layers:
                    x = F.relu(hidden_layer(x))
                x = torch.softmax(self.output(x), dim=-1)

                return x


        # Evaluates the quality of state-action pairs. Used for training the actor.
        class Critic(nn.Module):
                        
            def __init__(self, device, input_size, hidden_sizes, output_size, learning_rate):
                super(LunarLander.PPO.Critic, self).__init__()

                # Create the model
                self.input = nn.Linear(input_size, hidden_sizes[0]).to(device)
                self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]).to(device) for i in range(len(hidden_sizes)-1)])
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


        def __init__(self, device: torch.device, state_space: int, action_space: int,  hidden_sizes: list, learning_rate: float, gamma: float, epsilon: float, epochs: float, train=True):
            
            self.device = device
            self.state_space = state_space
            self.action_space = action_space
            self.hidden_sizes = hidden_sizes

            # Create a new actors and critics
            if train:

                # How quickly the current actor and critic update to match new data
                self.learning_rate = learning_rate

                # The discount factor on future rewards - changes how much we value expected future reward in the current state.
                self.gamma = gamma

                # The max amount of policy difference which will increase policy change (limits max policy change during updates)
                self.epsilon = epsilon

                # The number of times to retrain on an episode
                self.epochs = epochs

                # The loss function we use on the critic
                self.critic_loss = nn.MSELoss()

                # Randomly Initialize actor and critic networks
                """Takes in the state and outputs actions - the model's best guess about what actions will bring the greatest reward. Also called an "action policy"."""
                self.actor = LunarLander.PPO.Actor(device, state_space, hidden_sizes[0], action_space, learning_rate[0])
                """Takes in the state, and predicts the expected reward+discounted future reward this policy will attain by taking an action, which is called a Q(uality) value."""
                self.critic= LunarLander.PPO.Critic(device, state_space, hidden_sizes[1], 1, learning_rate[1])

        
        # Given a gamestate, find the best value for each continous action. Also return the action distribution for training later
        def take_action(self, state):
            state = Variable(state.unsqueeze(0)) 
            action_dist = Categorical(self.actor.forward(state).detach()) # Turn the ouput into a probability distribution
            action = action_dist.sample() # Randomly sample an action from the distribution
            action_logprob = action_dist.log_prob(action) # Store the logprob of that action to make the training calculations easier
            return action, action_logprob
        

        # Update the actor and critics, and then target actor/critic once the batch is done
        def update(self, episode_memories):

            # Find the cumulative reward and discounted future reward
            for index in range(len(episode_memories)-2, -1, -1): # Add the discounted future rewards in reverse order, and don't change the value of the last timestep
                episode_memories[index][3] += self.gamma * episode_memories[index+1][3]

            # Find the advantage for each timestep (the estimated value of specific actions in specific states, compared to the policy's normal value in that state)
            advantages = torch.zeros(len(episode_memories), dtype=torch.float, device=self.device)
            for index in range(len(episode_memories)):
                state = episode_memories[index][0]
                state_value = self.critic.forward(state).detach() # Don't do backprop on the advantage - we use the actor output instead
                state_action_value = episode_memories[index][3] # The Q value of the state-action pair
                advantages[index] = state_action_value - state_value # Advantage is how much better/worse an action in a state is, compared to a policy's normal Q value in that state.

            # Train multiple epochs since we don't adjust to large changes quickly
            indices = np.arange(len(episode_memories))
            for epoch in range(self.epochs):

                # Randomly shuffle the memories in the list to avoid bias
                np.random.shuffle(indices)

                # Train on each memory
                for memory_index in indices:

                    advantage = advantages[memory_index]
                    state = episode_memories[memory_index][0]
                    action = episode_memories[memory_index][1]
                    old_action_logprob = episode_memories[memory_index][2]
                    q_target = episode_memories[memory_index][3] # Reward + discounted future rewards == Q value

                    # Find the likelyhood of the current actor choosing the same action
                    action_dist = Categorical(self.actor.forward(state))
                    action_logprob = action_dist.log_prob(action)

                    # Update the actor
                    """Determine how different the actor is now in reference to this action in this state.
                    Use logprobs since they don't have the divide by 0 issue, and helps deal w/ low floating point value inaccuracies."""
                    policy_ratio = torch.exp(action_logprob - old_action_logprob) # This is the same as: action_logprob / old_action_lobgprob
                    """Entropy here helps our model explore - it punishes the model a bit for going all in one a single probability. It eventually should get outweighed
                    by the other part of the loss."""
                    policy_loss = -torch.min(policy_ratio*advantage, torch.clip(policy_ratio, 1-self.epsilon, 1+self.epsilon)*advantage) - 0.001*action_dist.entropy()
                    self.actor.optimizer.zero_grad()
                    policy_loss.backward()
                    self.actor.optimizer.step()

                    # Find what the current critic thinks the Q value of the state is
                    q_value = self.critic.forward(state)

                    # Update the critic
                    critic_loss = self.critic_loss(q_value, q_target) # MSE between the expected and actual q value
                    self.critic.zero_grad()
                    critic_loss.backward()
                    self.critic.optimizer.step()


    def __init__(self, device: torch.device, train: bool, param_dict: dict, verbose: bool=True):

        self.param_dict = param_dict

        self.device = device

        self.model_name = self.get_parameter("model_name", "")
        self.model_name = self.model_name + "_" if self.model_name else ""

        self.verbose = verbose

        # The clip amount - limits how much the model can change per training step
        self.epsilon = self.get_parameter("epsilon", LunarLander.DEFAULT_EPSILON)

        # How many episodes between renders
        self.render_freq = self.get_parameter("render_freq", None)

        # The number and size of hidden layers in the model
        self.hidden_sizes = LunarLander.DEFAULT_HIDDEN_SIZES
        self.hidden_sizes[0] = self.get_parameter("hidden_sizes", LunarLander.DEFAULT_HIDDEN_SIZES[0])

        # The number of games to train the model on, or we record it playing if testing
        if train:
            self.episodes = self.get_parameter("episodes", LunarLander.DEFAULT_EPISODES)
        else:
            self.episodes = self.get_parameter("episodes", 1)

        # The number of memories to accrue before training the model
        self.batch_size = self.get_parameter("batch_size", LunarLander.DEFAULT_BATCH_SIZE)

        # The number of times to re-train on each episode
        self.epochs = self.get_parameter("epochs", LunarLander.DEFAULT_EPOCHS)

        # How quickly the current actor and critic update to match new data
        self.learning_rate = self.get_parameter("learning_rate", LunarLander.DEFAULT_LEARNING_RATE)

        # The amount of weight places on future rewards
        self.gamma = self.get_parameter("gamma", LunarLander.DEFAULT_GAMMA)

        # The model our agent trains on
        self.env: gym.Env = gym.make(LunarLander.ENVIRONMENT_NAME, render_mode=None if train else "rgb_array", disable_env_checker=True)

        # The agent which interacts with and learns from the environment
        self.agent = LunarLander.PPO(self.device, self.env.observation_space.shape[0], self.env.action_space.n, self.hidden_sizes, self.learning_rate, 
                                     self.gamma, self.epsilon, self.epochs, train=train)

        # If the agent is in testing mode, load the saved model
        if not train:
            self.env = gym.wrappers.RecordVideo(self.env, os.getcwd() + LunarLander.SAVE_PATH, episode_trigger=lambda x: True, name_prefix="test_video")
            self.load_model()
            self.agent.actor.eval()


    # Create and train a DDPG model from scratch. Returns a dataframe containing the reward attained at each episode.
    def train(self) -> pd.DataFrame:

        # Record the reward, so that we can graph the changes in reward over time.
        reward_df = pd.DataFrame(columns=["episode", "reward"])

        # Track the number of consecutive wins to exit early
        total_win_count = 0
        consecutive_wins = 0
        total_recent_reward = -200*10
        recent_rewards = [-200]*10

        # Record the state and actions taken, and model outputs for each episode
        memories = []

        # Play the game for a number of episodes
        for episode in range(self.episodes):

            # Display the epsiode progress
            if self.verbose:
                print(f"Episode {episode+1}  :  Average Recent Reward: {total_recent_reward/10:.2f}  :  Total Wins: {total_win_count}      ", end='\r')

            # Only render episodes occasionally
            if self.render_freq:
                _render_mode = "human" if ((episode+1) % self.render_freq) == 0 else None
                self.env = gym.make(LunarLander.ENVIRONMENT_NAME, render_mode=_render_mode, disable_env_checker=True)

            # Reset the state of the environment so we can play it again
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device)

            # Make decisions each time-step of the game
            game_over: bool = False
            total_reward = 0
            while not game_over:

                # Select an action stochastically
                action, action_logprob = self.agent.take_action(state)

                # Take the action in the environment, and observe the new state
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
                total_reward += reward
                reward = torch.tensor([reward], dtype=torch.float, device=self.device)

                # Store the relevant training information in the memories buffer
                memories.append([state, action, action_logprob, reward])

                # Update the actor and critic several epochs using randomly sampled transitions from the last episode. The random-ness avoids bias and keeps it sample-efficient.
                if len(memories) > self.batch_size:
                    self.agent.update(memories)
                    memories = []

                # Check if the game is over
                game_over = terminated or truncated
                
                # If the game ended, record the total reward to plot later
                if game_over:
                    reward_df.loc[episode] = [episode, total_reward]

                # Now the next state is the current state
                state = next_state

            # Find the average recent score
            total_recent_reward -= recent_rewards.pop(0)
            total_recent_reward += total_reward
            recent_rewards.append(total_reward)
            
            # Check for a valid solution for the past 10 games
            if total_reward >= 200:
                total_win_count += 1
                consecutive_wins += 1
            else:
                consecutive_wins = 0
            if consecutive_wins ==  10:
                print(f"Solved the Environment on episode {episode}\n")
                break
                
        return reward_df


    # Load the final model from the previous training run, and dipslay it playing the environment
    def test(self) -> None:

        # Record the specified number of episodes
        for episode in range(self.episodes):

            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device)

            # Run the model until the environment ends
            game_over: bool = False
            while not game_over:
                
                # Record the video
                self.env.render()

                # Select an action stochastically
                action, action_logprob = self.agent.take_action(state)

                # Take the action in the environment, and observe the new state
                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)

                # Now the next state is the current state
                state = next_state

                # If the game is over, exit.
                game_over = terminated or truncated

        return
    

    def save_model(self):
        save_path = os.getcwd() + LunarLander.SAVE_PATH
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.agent.actor.state_dict(), save_path + '/' + self.model_name + LunarLander.ACTOR_MODEL_NAME + ".pt")


    def load_model(self):
        save_path = os.getcwd() + LunarLander.SAVE_PATH
        if not os.path.exists(save_path + '/' + self.model_name + LunarLander.ACTOR_MODEL_NAME + '.pt'):
            raise Exception("There are no saved models of the expected name. Run in training mode first with the same model-name parameter.")
        self.agent.actor = LunarLander.PPO.Actor(self.device, self.agent.state_space, self.agent.hidden_sizes[0], self.agent.action_space, self.learning_rate[0]).to(self.device)
        self.agent.actor.load_state_dict(torch.load(save_path + '/' + self.model_name + LunarLander.ACTOR_MODEL_NAME + '.pt'))