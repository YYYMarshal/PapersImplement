import sys
import os
import numpy as np
from utlis.config import Config
from utlis.default_config import default_config_maze, default_config_manipulation
from brhpo.launch import launch

if __name__ == '__main__':
    result_path = sys.argv[1]
    result_epoch = sys.argv[2] if len(sys.argv) > 2 else None
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Error, model file {result_path} not exists")
    args_path = result_path + '/' + 'config.log'
    args = Config().load_saved(args_path)
    model_path = args_path = result_path + '/model/'
    args.model_path = model_path
    args.model_i_epoch = result_epoch
    args.is_load = True

    agent = launch(args)
    agent.run_eval_render()
