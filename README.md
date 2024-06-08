# RL Gym Testing
 A collection of my attempts at solving different OpenAI Gym environments.

## How to Use
Models can be chosen by running "main.py" and calling the desired model.
You can choose to either train or test the model, and specify many parameters.

Ex: `python main.py --model Pendulum --mode train`

You can use the `--compare` option to obtain a graph displaying how changing a parameter changes the performance. 

Run `python main.py -h` for more information.

## Models
- DDPG Pendulum 
- DQN Mountain Car

## Pre-requisites 
Before running the model, you'll need to install the required packages. 
(I recommend setting up a virtual environment in this directory first)
You can do by running the following command in the main directory:
`pip install -r requirements.txt`
