import numpy as np
from utlis.config import ARGConfig
from utlis.default_config import default_config_maze, default_config_manipulation
from brhpo.launch import launch


def get_args():
    args = ARGConfig()
    # task and global parameters
    args.add_arg("env_name", "AntMaze", "Environment name")
    args.add_arg("seed", np.random.randint(0, 1000), "Random seed")
    args.add_arg("device", 0, "Computing device")
    args.add_arg("tag", "default", "task description")
    args.parser()

    return args


def main():
    config = get_args()
    if config.env_name in ["AntMaze", "AntPush", "AntFall"]:
        args = default_config_maze
        if config.env_name == "AntFall":
            args.subgoal_dim = 3
            args.h_action_dim = 3
    elif config.env_name in ["Reacher3D-v0", "Pusher-v0"]:
        args = default_config_manipulation
    else:
        raise "Unknown Environment"
    args.update(config)
    agent = launch(args)
    agent.run()


if __name__ == '__main__':
    main()
