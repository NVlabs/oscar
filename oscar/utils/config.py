# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import yaml
import json
from pathlib import Path

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch

from oscar.policies.robot_policy import CONTROLLER_MODE_MAPPING
from oscar.utils.config_utils import merge_nested_dicts
from oscar.utils.arg_utils import parse_arguments

from argparse import Namespace


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [Trace, Pour, Push]")


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def print_cfg(cfg):
    # Print config
    def default_handler(obj):
        if isinstance(obj, Namespace):
            return vars(obj)
        else:
            return str(obj.__class__)

    print()
    print(f"CONFIG")
    print("*" * 40)
    print(json.dumps(cfg, indent=4, default=default_handler))
    print("*" * 40)
    print()


def save_cfg(config, config_file):
    # Save appropriately based on file type
    config_type = config_file.split(".")[-1]
    if config_type == "json":
        return save_json(config=config, json_file=config_file)
    elif config_type == "yaml":
        return save_yaml(config=config, yaml_file=config_file)
    else:
        raise ValueError(f"Invalid config type specified: {config_type}")


def save_json(config, json_file):
    with open(json_file, 'w') as f:
        # preserve original key ordering
        json.dump(config, f, sort_keys=False, indent=4)


def save_yaml(config, yaml_file):
    # Make sure path exists
    out_dir = "/".join(yaml_file.split("/")[:-1])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(yaml_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


def load_cfg(args, use_rlg_config=False):
    # Make sure we're using rlg format
    assert use_rlg_config, "No support for anything other than using RL games currently!"

    # Load config
    with open(os.path.join(args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Optionally load and overwrite base cfg with additional config(s)
    if args.cfg_env_add is not None:
        for cfg_env_add in args.cfg_env_add:
            with open(os.path.join(cfg_env_add), 'r') as f:
                cfg_add = yaml.load(f, Loader=yaml.Loader)
                cfg = merge_nested_dicts(base_dict=cfg, extra_dict=cfg_add)

    cfg["args"] = args

    # Potentially add pretrained OSCAR model
    cfg["env"]["policy_controller"]["controller_config"]["delan"]["pretrained_model"] = None if \
        args.pretrained_delan in {None, "None"} else args.pretrained_delan

    task_name = cfg["env"]["task"]["name"]
    controller_type = cfg["env"]["policy_controller"]["controller_config"]["type"]
    name = args.experiment_name if args.experiment_name != 'Base' else cfg["policy"]["params"]["config"]['name']
    # Only save config if we're training
    if args.train:
        save_fpath = os.path.join(args.logdir, f"{task_name}/cfg", f"{task_name}_{controller_type}_{name}_cfg.yaml")
        save_cfg(cfg, config_file=save_fpath)
    print_cfg(cfg)

    env_cfg = cfg["env"]

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        env_cfg["task"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        env_cfg["task"]["episodeLength"] = args.episode_length

    # Optionally set num episdoes if testing
    if args.test:
        cfg["policy"]["params"]["config"]["player"]["games_num"] = args.num_test_episodes

    # Set physics domain randomization
    if "randomize" not in env_cfg["task"]:
        env_cfg["task"]["randomize"] = args.randomize
    else:
        env_cfg["task"]["randomize"] = args.randomize or env_cfg["task"]["randomize"]

    # Set deterministic mode
    if args.deterministic:
        cfg["deterministic"] = True

    logdir = args.logdir
    if use_rlg_config:
        exp_name = cfg["policy"]["params"]["config"]['name']
        if args.experiment_name != 'Base':
            exp_name = "{}".format(args.experiment_name)

        if env_cfg["task"]["randomize"]:
            exp_name += "_DR"

        # Override config name
        cfg["policy"]["params"]["config"]['name'] = exp_name

        if args.resume > 0:
            cfg["policy"]["params"]["load_checkpoint"] = True

        if args.checkpoint != "Base":
            cfg["policy"]["params"]["load_path"] = args.checkpoint

        # Set maximum number of training iterations (epochs)
        if args.max_iterations > 0:
            cfg["policy"]["params"]["config"]['max_epochs'] = args.max_iterations

        cfg["policy"]["params"]["config"]["num_actors"] = env_cfg["task"]["numEnvs"]

        seed = cfg.get("seed", 42)
        if args.seed > 0:
            seed = args.seed
        cfg["seed"] = seed

        # Update seed and determinstic settings in the policy
        cfg["policy"]["seed"] = cfg["seed"]
        cfg["policy"]["deterministic"] = cfg["deterministic"]

        # Set log directory
        cfg["policy"]["params"]["config"]["logdir"] = logdir

        # Compose info
        compose_cfg(args, cfg)

    else:
        # Override seed if passed on the command line
        if args.seed > 0:
            cfg["seed"] = args.seed

        log_id = args.logdir
        if args.experiment_name != 'Base':
            log_id = args.logdir + "_{}".format(args.experiment_name)
            if env_cfg["task"]["randomize"]:
                log_id += "_DR"

        logdir = os.path.realpath(log_id)
        os.makedirs(logdir, exist_ok=True)

        print_cfg(cfg)

    return cfg, logdir


def compose_cfg(args, cfg):
    """
    Composes a monolithic configuration from a partially-filled in one. Used for auto-filling values
    that are identical / can be inferred across multiple places.
    """
    # Unify controller info
    agent_cfg, policy_controller_cfg = cfg["env"]["agent"], cfg["env"]["policy_controller"]
    controller_type = policy_controller_cfg["controller_config"]["type"]
    if "dof_arm_mode" in agent_cfg and agent_cfg["dof_arm_mode"] == "__AUTO__":
        agent_cfg["dof_arm_mode"] = CONTROLLER_MODE_MAPPING[controller_type]
        if agent_cfg["dof_arm_mode"] == gymapi.DOF_MODE_POS:
            agent_cfg["dof_stiffness"] = agent_cfg["dof_stiffness_pos"]
        elif agent_cfg["dof_arm_mode"] == gymapi.DOF_MODE_EFFORT:
            agent_cfg["dof_stiffness"] = agent_cfg["dof_stiffness_effort"]
        else:
            raise ValueError("Only pos and effort control currently supported for robot arms!")
    # Overwrite stiffness and damping if we're using flex
    if cfg["env"]["sim"]["physics_engine"] == gymapi.SIM_FLEX:
        for i in len(agent_cfg["dof_stiffness"]):
            agent_cfg["dof_damping"][i] = 50.0
            if agent_cfg["dof_stiffness"][i] > 0:
                agent_cfg["dof_stiffness"][i] = 7000.0

    agent_cfg["denormalize_control"] = policy_controller_cfg["normalize_actions"]
    policy_controller_cfg["agent_config"] = agent_cfg
    policy_controller_cfg["n_envs"] = cfg["env"]["task"]["numEnvs"]
    policy_controller_cfg["device"] = 'cuda:0' if args.ppo_device == "GPU" else 'cpu'
    if policy_controller_cfg["control_freq"] is None:
        policy_controller_cfg["control_freq"] = round(1 / cfg["env"]["sim"]["dt"])
    policy_controller_cfg["control_steps_per_policy_step"] = \
        round(policy_controller_cfg["control_freq"] * cfg["env"]["sim"]["dt"])

    # Unify rlg config
    task_name = cfg["env"]["task"]["name"]
    cfg["policy"]["params"]["network"]["controller"] = policy_controller_cfg
    if cfg["policy"]["params"]["config"]["name"] == "__AUTO__":
        cfg["policy"]["params"]["config"]["name"] = f"{task_name}_{controller_type}"
    else:
        # We add the task / controller type to the name
        cfg["policy"]["params"]["config"]["name"] = f"{task_name}_{controller_type}_" + \
                                                    cfg["policy"]["params"]["config"]["name"]
    cfg["policy"]["params"]["config"]["env_name"] = "rlgpu"
    cfg["policy"]["params"]["config"]["env_config"] = cfg["env"]
    cfg["policy"]["params"]["config"]["num_actors"] = cfg["env"]["task"]["numEnvs"]

    # Determine whether to save video or not
    cfg["env"]["sim"]["save_video"] = args.save_video
    cfg["env"]["sim"]["no_force_sim_gpu"] = args.no_force_sim_gpu

    # cfg["policy"]["params"]["config"]["minibatch_size"] = \
    #     cfg["policy"]["params"]["config"]["num_actors"] * cfg["policy"]["params"]["config"]["steps_num"]


def parse_sim_params(args, cfg):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device == "GPU":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = (args.use_gpu or args.no_force_sim_gpu)
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    if args.device == "GPU":
        sim_params.use_gpu_pipeline = True

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False, use_rlg_config=False):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "action": "store_true", "default": False,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be Trace, Pour, or Push"},
        {"name": "--device", "type": str, "default": "GPU",
            "help": "Choose CPU or GPU device for running physics"},
        {"name": "--ppo_device", "type": str, "default": "GPU",
            "help": "Choose CPU or GPU device for inferencing PPO network"},
        {"name": "--no_force_sim_gpu", "action": "store_true", "default": False,
            "help": "If set, will infer sim engine device from @device arg, otherwise will run sim on GPU"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment_name", "type": str, "default": "Base"},
        {"name": "--cfg_train", "type": str,
            "default": "Base"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {"name": "--cfg_env_add", "type": str, "nargs": "+", "default": None, "help": "Optional additional cfg(s) to update @cfg_env. If multiple are specified, they were be processed in order"},
        {"name": "--seed", "type": int, "default": -1, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--deterministic", "action": "store_true", "default": False,
            "help": "Apply settings for more deterministic behaviour"},
        {"name": "--pretrained_delan", "type": str, "default": None,
            "help": "Absolute fpath to pretrained delan model if using OSCAR"},
        {"name": "--save_video", "action": "store_true", "default": False,
            "help": "Save videos during rollouts"},
        {"name": "--video_name", "type": str, "default": None,
            "help": "Custom video name when saving videos"},
        {"name": "--num_test_episodes", "type": int, "default": 10,
            "help": "How many test episodes to use during eval"},
    ]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
