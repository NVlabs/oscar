# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Set of utility functions for working with config files
"""

from isaacgym import gymapi, gymutil
from copy import deepcopy

def get_sim_params(sim_cfg):
    """
    Parses sim configuration dictionary to compose a corresponding SimParams object

    Returns:
        SimParams: Sim parameters composed from @sim_cfg
    """
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = sim_cfg["dt"]
    slices = sim_cfg["slices"] if sim_cfg["slices"] is not None else sim_cfg["subscenes"]
    physics_engine = sim_cfg["physics_engine"]
    sim_params.num_client_threads = slices

    # Determine whether we're using GPU
    use_gpu = "cpu" not in sim_cfg["device"] if isinstance(sim_cfg["device"], str) else sim_cfg["device"] >= 0

    if physics_engine == gymapi.SIM_FLEX:
        if use_gpu:
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = (use_gpu or not sim_cfg["no_force_sim_gpu"])
        sim_params.physx.num_subscenes = sim_cfg["subscenes"]
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    if use_gpu:
        sim_params.use_gpu_pipeline = True

    # Update defaults based on inputted sim config
    gymutil.parse_sim_config(sim_cfg, sim_params)

    return sim_params


def merge_nested_dicts(base_dict, extra_dict):
    """
    Iteratively updates @base_dict with values from @extra_dict. Note: This generates a new dictionary!

    Args:
        base_dict (dict): Nested base dictionary, which should be updated with all values from @extra_dict
        extra_dict (dict): Nested extra dictionary, whose values will overwrite corresponding ones in @base_dict

    Returns:
        dict: Updated dictionary
    """
    # Loop through all keys in @extra_dict and update the corresponding values in @base_dict
    base_dict = deepcopy(base_dict)
    for k, v in extra_dict.items():
        if k not in base_dict:
            base_dict[k] = v
        else:
            if isinstance(v, dict):
                if isinstance(base_dict[k], dict):
                    base_dict[k] = merge_nested_dicts(base_dict[k], v)
                else:
                    if base_dict[k] != v:
                        print(f"Different values for key {k}: {base_dict[k]}, {v}\n")
                    base_dict[k] = v
            else:
                if base_dict[k] != v:
                    print(f"Different values for key {k}: {base_dict[k]}, {v}\n")
                base_dict[k] = v

    # Return new dict
    return base_dict
