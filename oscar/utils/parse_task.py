# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import oscar
from oscar.tasks import *
from oscar.tasks.base.vec_task import VecTaskPython
from oscar.utils.config import warn_task_name


def parse_task(args, cfg, sim_params):
    # create native task and pass custom config
    if args.device == "CPU":
        print("Python CPU")
        sim_device = 'cpu'
        ppo_device = 'cuda:0' if args.ppo_device == "GPU" else 'cpu'
    else:
        print("Python GPU")
        sim_device = 'cuda:0'
        ppo_device = 'cuda:0'

    # Set the values in sim
    cfg["sim"]["physics_engine"] = args.physics_engine
    cfg["sim"]["headless"] = args.headless
    cfg["sim"]["device"] = sim_device

    try:
        task = eval(cfg["task"]["name"])(cfg=cfg)
    except NameError as e:
        warn_task_name()

    env = VecTaskPython(
        task=task,
        rl_device=ppo_device,
        clip_observations=cfg["task"]["clip_observations"],
        clip_actions=cfg["task"]["clip_actions"],
    )

    return task, env
