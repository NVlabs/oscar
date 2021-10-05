# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Set of utility functions for dealing with files
"""

import os
from pathlib import Path
from signal import signal, SIGINT
from sys import exit
import imageio

# Define callbacks
video_writer = None


def create_video_writer(name, fps=20):
    """
    Creates a video writer object
    Args:
        name (str): name to save video to. Should include .mp4
        fps (int): frames per second for playback
    Returns:
        video_writer: video writer to use
    """
    # Create video writer
    video_writer = imageio.get_writer(name, fps=fps)

    # Define handler
    def handler(signal_received, frame):
        # Handle any cleanup here
        print('SIGINT or CTRL-C detected. Closing video writer and exiting gracefully')
        video_writer.close()
        exit(0)

    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    # Return video writer object
    return video_writer


def verify_path(fpath):
    """
    Checks if a directory / file exists, if not, creates the appropriate directory
    Args:
        fpath (str): absolute directory / file path
    Returns:
        bool: True if it exists, else False (in the latter case, also creates the appropriate directory)
    """
    # Check if this directory exists
    exists = os.path.exists(fpath)
    # If it doesn't exist, we create this new directory
    if not exists:
        Path(fpath).mkdir(parents=True, exist_ok=True)

    return exists
