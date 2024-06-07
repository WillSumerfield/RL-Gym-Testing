"""
Choose which model to run, whether to test or train it, and what parameters to use.
"""

import os
import argparse
import torch
import agents
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# A list of all parameters passed into the param dictionary
PARAM_LIST = ["episodes", "hidden_sizes", "batch_size", "render_freq", "epsilon", "gamma", "tau"]


def CreateArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
                    prog='main.py',
                    description="Chooses a model to run,\n"
                                "whether to train it or test it,\n"
                                "and what parameters to run it with.",
                    epilog='',
                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', choices=["test", "train"], required=True, help="Whether the model should be trained, or a previously trained model should be loaded.")
    parser.add_argument('--model', choices=["Pendulum", "MountainCar"], required=True, help="What model should be run?")
    parser.add_argument('--compare', nargs=5, help="Only applicable if training. Check how the model changes as we vary a parameter.\n"
                        "You'll need to specify the parameter, the min parameter value, the max parameter value, and how many times we change the parameter and train.\n"
                        "Both the min and max are inclusive.\n"
                        "Ex: --compare gamma 0.8 1.0 10 8")
    parser.add_argument('--model-name', dest="model_name", default=None, help="The filename of the model to save/load, if training/testing (not include file extension).")
    parser.add_argument('--cpu', action='store_true', help="Use the CPU instead of a GPU.")

    # Parameters
    parser.add_argument('--episodes', type=int, default=None, help="How many episodes should the model train/test on? If not specified, a per-model default is used.")
    parser.add_argument('--hidden-sizes', dest="hidden_sizes", type=int, nargs='*', default=None, help="The number and size of each hidden layer. Will use model defaults if not specified.\n"
                        "Ex: --hidden_sizes 128 256 32    - This specifies that there are 3 hidden layers of those sizes.\n"
                        "Ensure that testing and training calls use the same number of hidden sizes.")
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=None, help="The size of training batches.")
    parser.add_argument('--render-freq', dest="render_freq", type=int, default=None, help="How many training episodes between renderings.\n"
                        "Only usable during training. Training is not rendered by default.")
    parser.add_argument('--epsilon', type=float, default=None, help="- Pendulum: The std. dev. of the gaussian noise applied to actions.\n"
                                                                    "- Mountain Car: The chance to choose a random action for exploration.")
    parser.add_argument('--learning-rate', dest="learning_rate", type=float, default=None, help="The amount to update the model based on learning from the data.")
    parser.add_argument('--gamma', type=float, default=None, help="The discount factor for future rewards.")
    parser.add_argument('--tau', type=float, default=None, help="The amount to update the target models.")

    return parser


# Returns if the argument errors are crticial and we should exit.
def CheckArgs(args: argparse.ArgumentParser) -> bool:
    if args.episodes is not None:
        if args.episodes < 0:
            print("Warning: Episodes less than 0, using model default.")
    if args.compare is not None:
        if not do_train:
            print("Comparison is only available when training. Either set the mode to training, or remove the comparison parameters.\n")
            return True
        if len(args.compare) != 5:
            print("Comparison requires five arguments:\n"
                  f"- parameter type: {PARAM_LIST}\n"
                  "- min parameter value\n"
                  "- max parameter value\n"
                  "- number of training instances\n"
                  "- number of runs to average over\n"
                  "Ex: --compare tau 1e-4 1e-2 5 5")
            return True
        if float(args.compare[3]) < 2:
            print("The number of comparison training instances must be at least two.")
            return True
        if not args.compare[0] in PARAM_LIST:
            print("The comparison parameter must be be one of the following:\n"
                  f"{PARAM_LIST}")
    if args.render_freq is not None and args.render_freq < 1:
        print("Render Frequency must be at least 1.")
        return True
    return False


# Choose which environment to run
if __name__ == "__main__":

    parser = CreateArgParser()
    args = parser.parse_args()

    do_train = args.mode == "train"

    # Check for invalid args
    if CheckArgs(args):
        exit()

    # Get the GPU device
    if not torch.cuda.is_available():
        print("CUDA not availabe or GPU not found.")
        exit()
    device = torch.device("cpu") if args.cpu else torch.device("cuda:0")

    # Create a dictionary of parameters to pass into the model
    param_dict = {"model_name": args.model_name,"episodes": args.episodes, "hidden_sizes": args.hidden_sizes, "batch_size": args.batch_size, "render_freq": args.render_freq,
                  "learning_rate": args.learning_rate, "gamma": args.gamma, "tau": args.tau, "epsilon": args.epsilon}

    # Find the chosen type of agent
    model_type = None
    if args.model == "Pendulum":
        model_type = agents.Pendulum
    elif args.model == "MountainCar":
        model_type = agents.MountainCar

    # Check if we picked a valid environment
    if model_type is None:
        print("\nInavalid Model.\n")
        parser.print_help()
        exit()

    # Train the agent
    if do_train:

        print("\nBeginning Training: \n")

        # If we're doing a comparison, run the training more than once with different args
        if (args.compare):
            parameter:str = args.compare[0]
            param_min:float = float(args.compare[1])
            param_max:float = float(args.compare[2])
            num_train:int = int(args.compare[3])
            num_runs:int = int(args.compare[4])

            # Run the model with different params each time
            print("Beginning Parameter Comparison:\n")
            print(f"Progress: 0/{num_train} = 0%", end='\r')
            rewards_df = pd.DataFrame(columns=["param_value", "episode", "reward"])

            # Save an untrained version of the model so that all the models start with the same base parameters
            model = model_type(device, train=True, param_dict=param_dict)
            model.save_model()

            # Try different values of the parameter
            for train_idx in range(num_train):

                reward_df = None # Holds the sum of all reward dfs
                
                # Train w/ each parameter a number of times, to get an average
                for run in range(num_runs):

                    # Set the parameter for this run
                    current_param_dict = param_dict
                    param_value = param_min + (param_max-param_min)/(num_train-1)*train_idx
                    param_dict[parameter] = param_value

                    # Create a new agent and load the untrained model
                    model = model_type(device, train=True, param_dict=param_dict, verbose=False)
                    model.load_model()

                    # Train the agent and save the reward for each episode of the current training run
                    run_df = model.train()

                    # Sum the rewards for each episode
                    if reward_df is not None:
                        reward_df["reward"] = reward_df["reward"] + run_df["reward"]
                    else:
                        reward_df = run_df

                # Find the average reward for each episode of the runs
                reward_df["reward"] /= num_runs

                # Append the rewards over episdoes to the reward dataframe
                reward_df["param_value"] = param_value
                rewards_df = reward_df if train_idx == 0 else pd.concat([rewards_df, reward_df], ignore_index=True)

                print(f"Progress: {train_idx+1}/{num_train} = {100*(train_idx+1)/num_train:.1f}%", end='\r')

            # Display the change in reward over time. 
            plot = sns.lineplot(x='episode', y='reward', hue='param_value', palette="Paired", data=rewards_df)
            pretty_param = parameter.replace("_", " ").capitalize()
            plt.title(f"Varying {pretty_param} Against Reward Over Episodes")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.savefig(f"{os.getcwd() + model.SAVE_PATH}/{model.model_name}_reward_across_{parameter}.png")
            plt.show()

        # Otherwise, just run the model once:
        else:

            model = model_type(device, train=True, param_dict=param_dict)
            reward_df = model.train()

            # Save the actor model when done training
            model.save_model()

            # Display the change in reward over time. 
            sns.lineplot(x='episode', y='reward', data=reward_df)
            plt.title(f"Reward Over Episodes")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.savefig(f"{os.getcwd() + model.SAVE_PATH}/{model.model_name}_reward_over_episodes.png")
            plt.show()

        print("\n\nFinished Training: \n")
    
    # Test the agent
    else:
        try:
            print("\nRunning Saved Model...\n")

            model = model_type(device, train=False, param_dict=param_dict)
            model.test()

            print("Finished Environment!\n")

        except Exception as e:
            print(f"\n{e}\n")
            parser.print_usage()
            print()