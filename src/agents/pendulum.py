"""
My attempt at solving a the pendulum problem with a Deep Determinist Policy Gradient based on the theory introduced in this paper:
https://arxiv.org/abs/1509.02971

DDPG Summary:
A DDPG is a 'Deep Deterministic Policy Gradient'. That means:
 - Deep = It uses Neural Networks
 - Deterministic = We choose the best action each time, excluding exploration (as opposed to stochastic, which chooses based on a probability distribution).
 - Policy Gradient = 

Credit to this article for general reference: 
https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
"""

import gym
import numpy as np
import os
import random
import pygame
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from .agent import Agent


class Pendulum(Agent):

    ENVIRONMENT_NAME = "Pendulum-v1"
    SAVED_MODELS_PATH = "/src/saved_models/pendulum"
    CRITIC_MODEL_NAME = "critic.pt"
    ACTOR_MODEL_NAME = "actor.pt"

    DISPLAY_SECONDS_PER_FRAME = 1/60
    DISPLAY_SIZE = (500, 500)


    # Transform the action space to be between 0 and 1. This makes the NN's work easier!
    class NormalizedEnv(gym.ActionWrapper):

        def action(self, action):
            act_k = (self.action_space.high - self.action_space.low)/ 2.
            act_b = (self.action_space.high + self.action_space.low)/ 2.
            return act_k * action + act_b

        def reverse_action(self, action):
            act_k_inv = 2./(self.action_space.high - self.action_space.low)
            act_b = (self.action_space.high + self.action_space.low)/ 2.
            return act_k_inv * (action - act_b)


    class DDPG():

        # Takes in actions as inputs, outputs an acation. Used to choose which actions to take during a run. This type of noise is called the "Ornstein-Ulhenbeck Process" process.
        class Actor(nn.Module):
            
            def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-4):
                super(Pendulum.DDPG.Actor, self).__init__()

                # Create the model
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, hidden_size)
                self.linear3 = nn.Linear(hidden_size, output_size)

                # Set the optimizer
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

            def forward(self, state) -> torch.Tensor:
                x = F.relu(self.linear1(state))
                x = F.relu(self.linear2(x))
                x = torch.tanh(self.linear3(x))

                return x


        # Evaluates the quality of state-action pairs. Used for training the actor.
        class Critic(nn.Module):
                        
            def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-3):
                super(Pendulum.DDPG.Critic, self).__init__()

                # Create the model
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, hidden_size)
                self.linear3 = nn.Linear(hidden_size, output_size)

                # Set the optimizer
                self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

                # The loss function
                self.loss = nn.MSELoss()


            def forward(self, state, action) -> torch.Tensor:
                x = torch.cat([state, action], 1)
                x = F.relu(self.linear1(x))
                x = F.relu(self.linear2(x))
                x = self.linear3(x)

                return x


        """Copies the parameters from the src model to the dst model. The tau parameter determines the proportion of the model which is copied over.
        When tau=1, the dst model is set equal to the src model."""
        def copy_model(source_model: nn.Module, destination_model: nn.Module, tau=1.0) -> None:
            for source_param, dest_param in zip(source_model.parameters(), destination_model.parameters()):
                dest_param.data.copy_(tau*source_param.data + (1.0-tau)*dest_param.data)


        def __init__(self, device: torch.device, state_space: int, action_space: int, hidden_size=256, gamma=0.99, tau=1e-2, load_best=False):
            
            self.device = device

            self.state_space = state_space
            self.action_space = action_space

            self.tau = tau

            # Load the best agent from the previous training set
            if load_best:
                
                # Load the saved actor and critic models
                model_path = os.getcwd() + Pendulum.SAVED_MODELS_PATH
                if not os.path.exists(model_path + '/' + Pendulum.CRITIC_MODEL_NAME) or not os.path.exists(model_path + '/' + Pendulum.ACTOR_MODEL_NAME):
                    raise Exception("There are no saved models. Run in training mode first.")
                    
                self.actor = Pendulum.DDPG.Actor(input_size=state_space, hidden_size=hidden_size, output_size=action_space).to(device)
                self.actor.load_state_dict(torch.load(model_path + '/' + Pendulum.ACTOR_MODEL_NAME))
                self.critic = Pendulum.DDPG.Critic(input_size=state_space+action_space, hidden_size=hidden_size, output_size=action_space).to(device)
                self.critic.load_state_dict(torch.load(model_path + '/' + Pendulum.CRITIC_MODEL_NAME))

            # Create a new actors and critics
            else:

                # Randomly Initialize actor and critic networks
                """Takes in states and outputs actions - the model's best guess about what actions will bring the greatest reward. Also called an "action policy"."""
                self.actor = Pendulum.DDPG.Actor(input_size=state_space, hidden_size=hidden_size, output_size=action_space)
                self.actor = self.actor.to(device)
                """Takes in states and actions, and predicts the expected reward for each action, which is called a Q value."""
                self.critic = Pendulum.DDPG.Critic(input_size=state_space+action_space, hidden_size=hidden_size, output_size=action_space)
                self.critic = self.critic.to(device)

                # Create target actor and target critic networks. In the beginning, they are copies of the actor and critic
                """The actor we train, but don't use for decision making. We set the actor equal to the target actor regularly 
                (but not every sample, or the model would change be unstable and change super fast), which ensures that our decisions are relativley up to date."""
                self.target_actor = Pendulum.DDPG.Actor(input_size=state_space, hidden_size=hidden_size, output_size=action_space)
                self.target_actor = self.target_actor.to(device)
                Pendulum.DDPG.copy_model(self.actor, self.target_actor)
    
                """The critic we train, but don't use for evaluating state/action pairs. We set the critic equal to the target critic regularly 
                (but not every sample, or the model would change be unstable and change super fast), which ensures that our decisions are relativley up to date."""
                self.target_critic = Pendulum.DDPG.Critic(input_size=state_space+action_space, hidden_size=hidden_size, output_size=action_space)
                self.target_critic = self.target_critic.to(device)
                Pendulum.DDPG.copy_model(self.critic, self.target_critic)

                # The discount factor on future rewards - changes how much we value expected future reward in the current state.
                self.gamma = gamma

        
        # Given a gamestate, find the best value for each continous action.
        def take_action(self, state):
            state = Variable(torch.from_numpy(state).to(self.device).float().unsqueeze(0))
            action = self.actor.forward(state)
            action = action.detach().cpu().numpy()[0, 0] # Convert the tensor to a numpy array
            return action
        

        # Update target actor and critics, and then set the current actor 
        def update(self, memory_batch):

            # Put the memories into the pytorch tensors
            states = np.empty((0, self.state_space))
            actions = np.empty((0, self.action_space))
            rewards = np.empty((0, 1))
            next_states = np.empty((0, self.state_space))

            for state, action, reward, next_state in memory_batch:
                states = np.vstack((states, state))
                actions = np.vstack((actions, action))
                rewards = np.vstack((rewards, reward))
                next_states = np.vstack((next_states, next_state))
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)

            # Find the current critic by minimizing the loss (Comparing the expected Q value to the actual Q value, and doing backprop)
            q_values = self.critic.forward(states, actions)
            next_actions = self.target_actor.forward(next_states)
            q_prime_values = rewards + self.gamma * self.target_critic.forward(next_states, next_actions.detach())
            critic_loss = self.critic.loss(q_values, q_prime_values)

            # Find current actor and critic's average Q values 
            current_actions = self.actor.forward(states)
            q_values = self.critic.forward(states, current_actions)
            policy_loss = -q_values.mean()

            # Perform backpropigation on the policy loss to update the target actor
            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            self.actor.optimizer.step()

            # Perform backpropigation on the critic network using the critic loss
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step

            # Update the target networks
            Pendulum.DDPG.copy_model(self.actor, self.target_actor, tau=self.tau)
            Pendulum.DDPG.copy_model(self.critic, self.target_critic, tau=self.tau)
            

        def save_model(self):
            model_path = os.getcwd() + Pendulum.SAVED_MODELS_PATH
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.target_actor.state_dict(), model_path + '/' + Pendulum.ACTOR_MODEL_NAME)
            torch.save(self.target_critic.state_dict(), model_path + '/' + Pendulum.CRITIC_MODEL_NAME)


    def __init__(self, device: torch.device, train: bool, episodes: int=50, batch_size: int=128):

        self.device = device

        # Train the model
        if train:

            # The number of games to train the model on
            self.episodes = episodes 

            """The replay buffer holds memories of all past transitions. We essentially delay the training to make this an 'off policy' model.
            If the agent explored 'on-policy', it might not explore enough since this model is deterministic instead of stochastic."""
            self.replay_buffer = []

            # The number of memories to train the model on each update.
            self.batch_size = batch_size


    def train(self) -> None:
        
        # Create the environment
        env: gym.Env = Pendulum.NormalizedEnv(gym.make(Pendulum.ENVIRONMENT_NAME, disable_env_checker=True))

        # Create the agent
        agent = Pendulum.DDPG(self.device, env.observation_space.shape[0], env.action_space.shape[0])

        # Record the reward, so that we can graph the changes in reward over time.
        total_rewards = np.array([])

        # Play the game for a number of episodes
        for episode in range(self.episodes):

            # Display the epsiode progress
            print(f"Episode: {episode+1}/{self.episodes} = {100*(episode+1)/self.episodes}%", end='\r')

            # Reset the environment
            state, info = env.reset()

            # Make decisions each time-step of the game
            game_over: bool = False
            while not game_over:

                # Select an action deterministically, but use a little noise to do exploration.
                action = agent.take_action(state)
                action += np.random.normal(0, 0.1)

                # Take the action in the environment, and observe the new state
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = np.array(next_state)
                reward = np.array(reward)

                # Check if the game is over
                game_over = terminated or truncated
                
                # If the game ended, record the total reward to plot later
                if game_over:
                    total_rewards = np.append(total_rewards, reward)

                # Store the 'transition (current state, current action, achieved reward, new state) in the Replay Buffer
                self.replay_buffer.append([state, action, reward, next_state])

                # Update the actor and critic using the whole replay buffer. Shuffle it to stabalize the learning
                if (len(self.replay_buffer) > self.batch_size):
                    agent.update(random.sample(self.replay_buffer, self.batch_size))

                # Now the next state is the current state
                state = next_state
                

        # Save the actor and critic models when done training
        agent.save_model()

        # Tell the user we're done and the model is saved
        print("\n\nFinished training and saved the model.\n")
        
        # Display how the reward changed over the epiodes
        sns.lineplot(x=np.arange(len(total_rewards)), y=total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    # Load the final model from the previous training run, and dipslay it playing the environment
    def test(self) -> None:

        # Create the environment
        env = Pendulum.NormalizedEnv(gym.make(Pendulum.ENVIRONMENT_NAME, render_mode="rgb_array", disable_env_checker=True))
        state, info = env.reset()

        # Load the model
        agent = Pendulum.DDPG(self.device, env.observation_space.shape[0], env.action_space.shape[0], load_best=True)

        # Initialize the display
        pygame.init()
        display = pygame.display.set_mode(self.DISPLAY_SIZE, 0, 32)
        clock = pygame.time.Clock()
        pygame.display.flip()

        # Let the user know that the game has begun
        print("\nPlaying the game with the best model...")

        # Run the model until the environment 
        game_over: bool = False
        while not game_over:

            # Select an action. Only select the best one here, since we're showing off the model's capabilities
            action = agent.take_action(state)

            # Take the action in the environment, and observe the new state
            next_state, reward, terminated, truncated, info = env.step(action)

            # Display the game
            image = env.render()
            image = Image.fromarray(image, 'RGB')
            image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
            display.blit(image, (0, 0))
            pygame.display.update()

            # Set the FPS at 60
            clock.tick(60)

            # If the game is over, exit.
            if terminated or truncated:
                print("\nFinished the game!")
                return

        return
    