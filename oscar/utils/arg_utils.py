# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
from isaacgym import gymapi

# parses all of the common Gym example arguments and returns them to the caller
# note that args.physics_engine stores the gymapi value for the desired physics engine
def parse_arguments(description="OSCAR Example", headless=False, no_graphics=False, custom_parameters=[]):
    parser = argparse.ArgumentParser(description=description)
    if headless:
        parser.add_argument('--headless', action='store_true', help='Run headless without creating a viewer window')
    if no_graphics:
        parser.add_argument('--nographics', action='store_true',
                            help='Disable graphics context creation, no viewer window is created, and no headless rendering is available')
    parser.add_argument('--compute_device_id', type=int, default=0, help='Physics Device ID')
    parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')

    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument('--flex', action='store_true', help='Use FleX for physics')
    physics_group.add_argument('--physx', action='store_true', help='Use PhysX for physics')
    physics_group.add_argument('--physx_gpu', action='store_true', help='Use PhysX GPU for physics')

    parser.add_argument('--num_threads', type=int, default=0, help='Number of cores used by PhysX')
    parser.add_argument('--subscenes', type=int, default=0, help='Number of PhysX subscenes to simulate in parallel')
    parser.add_argument('--slices', type=int, help='Number of client threads that process env slices')

    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            if "help" not in argument:
                argument["help"] = ""

            if "type" in argument:
                assert "default" in argument, "ERROR: default must be specified if using type"

            # Create argument
            name = argument.pop("name")
            parser.add_argument(name, **argument)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()

    # Set the proper gymapi SIM_ value for physics engine
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = False  # for PhysX only, Flex always use GPU
    if args.flex:
        args.physics_engine = gymapi.SIM_FLEX

    if args.physx_gpu:
        args.physics_engine = gymapi.SIM_PHYSX
        args.use_gpu = True

    # Using --nographics implies --headless
    if no_graphics and args.nographics:
        args.headless = True

    if args.slices is None:
        args.slices = args.subscenes

    return args
