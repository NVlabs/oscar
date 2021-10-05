# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from .franka import Franka
from .franka_gripper import FrankaGripper
from .franka_pitcher import FrankaPitcher

# Define mappings from agent name to agent class
AGENT_MAPPING = {
    "franka": Franka,
    "franka_gripper": FrankaGripper,
    "franka_pitcher": FrankaPitcher,
}

AGENTS = set(AGENT_MAPPING.keys())
