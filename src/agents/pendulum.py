"""
My attempt at solving a the pendulum problem with a Deep Determinist Policy Gradient based on the theory introduced in this paper:
https://arxiv.org/abs/1509.02971

DDPG Summary:
A DDPG is a 'Deep Deterministic Policy Gradient'. That means:
 - Deep = It uses Neural Networks
 - Deterministic = We choose the best action each time, excluding exploration (as opposed to stochastic, which chooses based on a probability distribution).
 - Policy Gradient = We follow the gradient of the policy (backpropigation) to find the optimal solution!

These models are useful because they can ouput continous actions, rather than simply choosing the best action from a finite set!
It is also "Off-Policy", which means that it updates the model based on expected future actions, but not necessarily the actions that the model ended up picking!
This is good because it allows our goal to remain "Find the optimal solution", but our agent gets to explore and take bad actions.

DDPGs have four models - an actor and a critic, and a target actor and target critic.
The actor is what plays the game - it explores the environment and tries to earn reward. 
We use the critic to evaluate the actor - this is what lets us know how to improve the actor. The critic tries to learn what actions are good/bad in what states,
based the reward recieved (and the expected reward from the next state that will put us in). We use these educated guesses about the Quality (Q values = Quality estimate) 
to tell the actor what it does well, and what it should do differently. 

The target actor and critic are only used in the training process. They are meant to be more stable, more correct versions than the actor and critic. We only
update them a little bit based on changes we make to the actor and critic, which is why they're stable - we don't change them too much too fast. So why don't we use
these models for the actual acting and critiquing? That's because we *like* a little instability in our actor and critic. We want them to explore and not get hung up on
pre-conceived ideas about what action/states are good or bad. 

We also use a "Relay Buffer", which means that we record every state, action, reward, and next state (called a transition), and then train the models at the end of 
each step of the game (not each episode! Every single step!). We also don't remove items from the buffer when we do this, so we re-use old samples. A big reason for using
this is to break correlation beteween consecutive transitions. This pendulum game has a lot of really similar states which all happen in series (imagine how similar the states
are when the pendulum is slowly gaining speed to swing up). This can create odd behaviors as the model over-attunes to those transitions. Keeping the transitions out of order
can fix that issue. 

To go into more detail about how we use the target actor and critic:
They only used when taking our best guess at what the true value of an action/state pair is. We use this to update the critic. When we're figuring out how good the critic is,
we first look at the reward we gained from a transiton. Then we take our best guess at what the value of the next state would be, by using the target actor to pick what 
action it would take in that situation, and then the target critic evaluates the move that the target actor just took. We compare that 'target Q value' to the Q value of 
the normal critic. We essentially assume that the target actor and critic are perfect, and change the current critic based off of the loss between those values. 
The target actor and target critic aren't perfect of course, but they are generally better. If we do this enough, the model improves and should eventually converge on
a good solution!

I opted for using a simple gaussian noise function to help the model explore. I simply bump the output action of the actor up or down a little bit w/ guassian noise.

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
    ACTOR_MODEL_NAME = "actor.pt"

    DISPLAY_SECONDS_PER_FRAME = 1/60
    DISPLAY_SIZE = (500, 500)


    # Transform the action space to be between -1 and 1. This makes the NN's work easier!
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

        # Takes in actions as inputs, outputs an acation. Used to choose which actions to take during a run.
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


        def __init__(self, device: torch.device, state_space: int, action_space: int, hidden_size=256, gamma=0.99, tau=1e-2, test=False):
            
            self.device = device
            self.state_space = state_space
            self.action_space = action_space

            # Load the best agent from the previous training set
            if test:
                
                # Load the saved actor and critic models
                model_path = os.getcwd() + Pendulum.SAVED_MODELS_PATH
                if not os.path.exists(model_path + '/' + Pendulum.ACTOR_MODEL_NAME):
                    raise Exception("There are no saved models. Run in training mode first.")
                self.actor = Pendulum.DDPG.Actor(input_size=state_space, hidden_size=hidden_size, output_size=action_space).to(device)
                self.actor.load_state_dict(torch.load(model_path + '/' + Pendulum.ACTOR_MODEL_NAME))

            # Create a new actors and critics
            else:

                # Randomly Initialize actor and critic networks
                """Takes in states and outputs actions - the model's best guess about what actions will bring the greatest reward. Also called an "action policy"."""
                self.actor = Pendulum.DDPG.Actor(input_size=state_space, hidden_size=hidden_size, output_size=action_space).to(device)
                """Takes in states and actions, and predicts the expected reward for each action (just one in our case), which is called a Q(uality) value."""
                self.critic = Pendulum.DDPG.Critic(input_size=state_space+action_space, hidden_size=hidden_size, output_size=action_space).to(device)

                # Create target actor and target critic networks. In the beginning, they are copies of the actor and critic
                """The actor we train, but don't use for decision making. We update this whenever we update the normal actor, but just a little bit.
                This ensures that the target actor remains up to date, but isn't as 'jumpy' as the regular actor. This actions we get from this model are 
                what makes DDPG an off-policy model."""
                self.target_actor = Pendulum.DDPG.Actor(input_size=state_space, hidden_size=hidden_size, output_size=action_space).to(device)
                Pendulum.DDPG.copy_model(self.actor, self.target_actor)
    
                """The critic we train, but don't use for updating the actor. We update this whenever we update the normal critic, but just a little bit.
                This ensures that the target critic remains up to date, but isn't as 'jumpy' as the regular critic. This Q values we get from this model are 
                what makes DDPG an off-policy model."""
                self.target_critic = Pendulum.DDPG.Critic(input_size=state_space+action_space, hidden_size=hidden_size, output_size=action_space).to(device)
                Pendulum.DDPG.copy_model(self.critic, self.target_critic)

                # The discount factor on future rewards - changes how much we value expected future reward in the current state.
                self.gamma = gamma

                # The rate at which the target models are adjusted when we update the current models
                self.tau = tau

        
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
            next_q_values = self.target_critic.forward(next_states, next_actions.detach())
            q_prime_values = rewards + self.gamma * next_q_values
            critic_loss = self.critic.loss(q_values, q_prime_values)

            """Find current actor's best action, and use that to find the critic's Q value for that state action pair. 
            The 'loss' in our case is just the opposite of the reward. Our Q value is an estimate of reward, so we can just get the negative of that.
            Since we're evaluating a whole batch at once, we can find the mean Q value and perform backprop on that."""
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
            self.critic.optimizer.step()

            # Update the target networks a little based on the current models
            Pendulum.DDPG.copy_model(self.actor, self.target_actor, tau=self.tau)
            Pendulum.DDPG.copy_model(self.critic, self.target_critic, tau=self.tau)
            

        def save_model(self):
            model_path = os.getcwd() + Pendulum.SAVED_MODELS_PATH
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(self.target_actor.state_dict(), model_path + '/' + Pendulum.ACTOR_MODEL_NAME)


    def __init__(self, device: torch.device, train: bool, episodes: int=50, noise: float=0.05, batch_size: int=128, render_freq: int=-1):

        self.device = device
        self.noise = noise

        # How many episodes between renders
        self.render_freq = render_freq

        # Initialize the display
        if self.render_freq > 0 or not train:
            self.do_render = True
            pygame.init()
            self.display = pygame.display.set_mode(self.DISPLAY_SIZE, 0, 32)
            self.clock = pygame.time.Clock()
            pygame.display.flip()
        else:
            self.do_render = False

        # Train the model
        if train:

            # The number of games to train the model on
            self.episodes = episodes 

            """The replay buffer holds memories of all past transitions. We sample this randomly to update the model, which helps avoid redundancy and bias
            introduced by sequential transitions which are super similar."""
            self.replay_buffer = []

            # The number of memories to train the model on each update.
            self.batch_size = batch_size


    def train(self) -> None:
        
        # Create the environment
        env: gym.Env = Pendulum.NormalizedEnv(gym.make(Pendulum.ENVIRONMENT_NAME, render_mode="rgb_array", disable_env_checker=True))

        # Create the agent
        agent = Pendulum.DDPG(self.device, env.observation_space.shape[0], env.action_space.shape[0])

        # Record the reward, so that we can graph the changes in reward over time.
        total_rewards = np.array([])

        # Play the game for a number of episodes
        for episode in range(self.episodes):

            # Display the epsiode progress
            print(f"Episode: {episode+1}/{self.episodes} = {100*(episode+1)/self.episodes}%", end='\r')

            # Reset the state of the environment so we can play it again
            state, info = env.reset()

            # Only render episodes occasionally
            render_episode = self.do_render and (episode % self.render_freq == 0)

            # Make decisions each time-step of the game
            game_over: bool = False
            while not game_over:

                # Select an action deterministically, but use a little noise to do exploration.
                action = agent.take_action(state)
                action += np.random.normal(0, self.noise)

                # Take the action in the environment, and observe the new state
                next_state, reward, terminated, truncated, info = env.step(np.array([action]))
                next_state = np.array(next_state)
                reward = np.array(reward)

                # Display the game
                if render_episode:
                    image = env.render()
                    image = Image.fromarray(image, 'RGB')
                    image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
                    self.display.blit(image, (0, 0))
                    pygame.display.update()

                    # Set the FPS at 60
                    self.clock.tick(60)

                # Store the 'transition (current state, current action, achieved reward, new state) in the Replay Buffer
                self.replay_buffer.append([state, action, reward, next_state])

                # Check if the game is over
                game_over = terminated or truncated
                
                # If the game ended, record the total reward to plot later
                if game_over:
                    total_rewards = np.append(total_rewards, reward)

                # Update the actor and critic using the a random sample from the replay buffer. The random-ness avoids bias and keeps it sample-efficient.
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
        agent = Pendulum.DDPG(self.device, env.observation_space.shape[0], env.action_space.shape[0], test=True)

        # Let the user know that the game has begun
        print("\nPlaying the game with the best model...")

        # Run the model until the environment 
        game_over: bool = False
        while not game_over:

            # Select an action. Only select the best one here, since we're showing off the model's capabilities
            action = agent.take_action(state)

            # Take the action in the environment, and observe the new state
            next_state, reward, terminated, truncated, info = env.step(action)

            # Now the next state is the current state
            state = next_state

            # Display the game
            image = env.render()
            image = Image.fromarray(image, 'RGB')
            image = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
            self.display.blit(image, (0, 0))
            pygame.display.update()

            # Set the FPS at 60
            self.clock.tick(60)

            # If the game is over, exit.
            if terminated or truncated:
                print("\nFinished the game!")
                return

        return
    