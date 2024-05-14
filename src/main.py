"""
Choose which model to run, and whether to train it, or watch the best version.
"""

import argparse
import torch
import agents


def CreateArgParser():
    parser = argparse.ArgumentParser(
                    prog='main.py',
                    description="Chooses an environment to run.\n"
                                 "Environments are:\n"
                                 " - Pendulum\n",
                    epilog='',
                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('environment')
    parser.add_argument('-t', '--train', action='store_true')

    return parser


# Choose which environment to run
if __name__ == "__main__":

    parser = CreateArgParser()
    args = parser.parse_args()

    # Get the GPU device
    if not torch.cuda.is_available():
        print("CUDA not availabe or GPU not found.")
        exit()
    device = torch.device("cuda:0")

    # Find the right agent
    agent: agents.Agent = None
    if args.environment == "Pendulum":
        agent = agents.Pendulum(device, args.train, render_freq=10)

    # Check if we picked a valid environment
    if agent is None:
        print("\nInavalid Environment.\n")
        parser.print_help()
        exit()

    # Either train the agent, or display a run of the best agent
    if args.train:
        print("\nBeginning Training: \n")
        agent.train()
    else:
        try:
            agent.test()
        except Exception as e:
            print(f"\n{e}\n")
            parser.print_help()
            print()