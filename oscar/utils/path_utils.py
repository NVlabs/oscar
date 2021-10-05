# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Utilities for generating paths in free space
"""
import torch
import xml.etree.ElementTree as ET
from oscar.utils.matplotlib_utils import parse_path
from oscar.utils.torch_utils import quat_mul, axisangle2quat, quat_slerp
from oscar import ASSETS_ROOT
import os


# Define constant
_PI = 3.1415926536


class TrajectoryPath:
    """
    Basic class for generating a parameterized path.

    Args:
        center_pos (3-array): Center (x,y,z) position for the desired loop
        center_quat (4-array): Center (x,y,z,w) quaternion rotation for the desired path
        center_pos_noise (float): Amount of noise (+/- this value) to apply when generating center path position
        center_rot_noise (float): Amount of noise (+/- this value) to apply when generating center path orientation
        n_steps (int): Number of steps to generate in the path
        n_paths (int): How many paths to generate at a single time
        device (str): Which device to send all tensor to
        kwargs (dict): Not used; dummy var to sink extraneous variables (will print warning)
    """
    def __init__(
        self,
        center_pos=(0, 0, 0),
        center_quat=(0, 0, 0, 1),
        center_pos_noise=0.05,
        center_rot_noise=0.5,
        n_steps=100,
        n_paths=1,
        device="cuda:0",
        **kwargs,
    ):
        # Print warning if kwargs is not empty
        for k, v in kwargs.items():
            print(f"Warning: Got unexpected key {k} in Path args!")

        # Store number of paths and steps
        self.n_steps = n_steps
        self.n_paths = n_paths

        # Store counters for paths
        self.current_step = torch.zeros(n_paths, dtype=torch.long, device=device)

        # Store values for generating center pose
        self.center_pos = center_pos
        self.center_quat = center_quat
        self.center_pos_noise = center_pos_noise
        self.center_rot_noise = center_rot_noise
        self.sampled_center_pos = torch.tensor([center_pos] * n_paths, dtype=torch.float, device=device)
        self.sampled_center_quat = torch.tensor([center_quat] * n_paths, dtype=torch.float, device=device)

        # Store device
        self.device = device

        # Reset paths
        self.reset()

    def reset(self, path_ids=None):
        """
        Resets the paths specified by @path_ids. If None, we reset all paths

        Args:
            path_ids (None or tensor): If specified, should be the specific ids corresponding to the paths to regenerate
        """
        if path_ids is None:
            path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        # Update poses
        self._update_center_pose(path_ids=path_ids)

        # Current steps default back to 0
        self.current_step[:] = 0

    def _update_center_pose(self, path_ids):
        """
        Resamples all center positions and quaternions
        Args:
            path_ids (None or tensor): If specified, should be the specific ids corresponding to the path poses
                to update
        """
        n_paths = len(path_ids)
        # Generate center locations
        pos_noise_dir = torch.normal(0., 1., size=(n_paths, 3), device=self.device)
        pos_noise_mag = self.center_pos_noise * \
            (-1. + 2. * torch.rand(n_paths, 1, dtype=torch.float, device=self.device))
        pos_noise = pos_noise_mag * pos_noise_dir / torch.norm(pos_noise_dir, dim=-1, keepdim=True)

        rot_noise_dir = torch.normal(0., 1., size=(n_paths, 3), device=self.device)
        rot_noise_mag = self.center_rot_noise * \
            (-1. + 2. * torch.rand(n_paths, 1, dtype=torch.float, device=self.device))
        rot_noise = axisangle2quat(rot_noise_mag * rot_noise_dir / torch.norm(rot_noise_dir, dim=-1, keepdim=True))

        self.sampled_center_pos[path_ids] = \
            torch.tensor([self.center_pos] * n_paths, dtype=torch.float, device=self.device) + pos_noise
        self.sampled_center_quat[path_ids] = \
            quat_mul(torch.tensor([self.center_quat] * n_paths, dtype=torch.float, device=self.device), rot_noise)

    def _generate_pose(self, idx):
        """
        Generates the current step in the path as specified by @idx.

        NOTE: This should be implemented by subclass as a private method, and
        should NOT take center pos / rot into consideration! (i.e.: should be centered around (0,0,0) with
            quat (0,0,0,1)

        Args:
            idx (tensor): Indexes for each parameterized path to grab location

        Returns:
            2-tuple:
                - tensor: (x,y,z) cartesian position of the location
                - tensor: (x,y,z,w) quaternion of the location
        """
        raise NotImplementedError

    def generate_pose(self, idx=None):
        """
        Generates the current step in the path as specified by @idx.

        Args:
            idx (None or int or tensor): If specified, index(es) for each parameterized path to grab location

        Returns:
            2-tuple:
                - tensor: (x,y,z) cartesian position of the location
                - tensor: (x,y,z,w) quaternion of the location
        """
        if isinstance(idx, int):
            idx = torch.tensor([idx] * self.n_paths, dtype=torch.int, device=self.device)

        pos, quat = self._generate_pose(idx=idx if idx is not None else self.current_step)

        # If index is None, we increment the counter
        if idx is None:
            self.current_step += 1

        # Postprocess values and return
        return self._postprocess_pos(pos), self._postprocess_quat(quat)

    def _postprocess_pos(self, pos):
        """
        Postprocesses @pos by offsetting the value by self.center_pos

        Args:
            pos (tensor): (n_paths, 4) tensor where fianl dim is (x,y,z) cartesian position tensor to
                offset by self.center_pos

        Returns:
            tensor: offset pos tensor
        """
        return pos + self.sampled_center_pos

    def _postprocess_quat(self, quat):
        """
        Postprocesses @quat by rotating the value by self.center_rot

        Args:
            quat (tensor): (n_paths, 4) tensor where final dim is (x,y,z,w) tensor to rotate by self.center_rot

        Returns:
            tensor: rotated ori tensor
        """
        return quat_mul(quat, self.sampled_center_quat)


class CirclePath(TrajectoryPath):
    """
    A path consisting of a single circle

    Args:
        radius (float): Radius of the generated circle
        radius_noise (float): Amount of noise to vary radius by per path generated
        plane (2-tuple): Axes that form the plane upon which the circle path will be generated. Options are {x, y, z}
        randomize_axes (bool): If True, will randomize the axes for the circle path between resets
        center_pos (3-array): Center (x,y,z) position for the desired loop
        center_quat (4-array): Center (x,y,z,w) quaternion rotation for the desired path
        center_pos_noise (float): Amount of noise (+/- this value) to apply when generating center path position
        center_rot_noise (float): Amount of noise (+/- this value) to apply when generating center path orientation
        circle_tilt_noise (float): Amount of noise (+/- this value) to apply when generating circle path orientation
        n_steps (int): Number of steps to generate in the path
        n_paths (int): How many paths to generate at a single time
        device (str): Which device to send all tensor to
        kwargs (dict): Not used; dummy var to sink extraneous variables (will print warning)
    """
    def __init__(
        self,
        radius=0.5,
        radius_noise=0.1,
        plane=("x", "z"),
        randomize_axes=False,
        center_pos=(0, 0, 0),
        center_quat=(0, 0, 0, 1),
        center_pos_noise=0.05,
        center_rot_noise=0.5,
        circle_tilt_noise=0.785,
        n_steps=100,
        n_paths=1,
        device="cuda:0",
        **kwargs,
    ):
        # Store radius info
        self.radius = radius
        self.radius_noise = radius_noise
        self.sampled_radius = torch.tensor([radius] * n_paths, dtype=torch.float, device=device)

        # Store plane info
        MAPPING = {val: i for i, val in enumerate(("x", "y", "z"))}
        self.sampled_axes = torch.zeros(n_paths, 3, dtype=torch.long, device=device)
        self.sampled_axes[:, 0] = MAPPING.pop(plane[0])
        self.sampled_axes[:, 1] = MAPPING.pop(plane[1])
        self.sampled_axes[:, 2] = list(MAPPING.values())[0]
        self.randomize_axes = randomize_axes
        self.circle_tilt_noise = circle_tilt_noise
        self.sampled_circle_tilt = torch.zeros(n_paths, dtype=torch.float, device=device)

        # Run super init first
        super().__init__(
            center_pos=center_pos,
            center_quat=center_quat,
            center_pos_noise=center_pos_noise,
            center_rot_noise=center_rot_noise,
            n_steps=n_steps,
            n_paths=n_paths,
            device=device,
            **kwargs,
        )

    def reset(self, path_ids=None):
        """
        In addition to normal resetting, update the radius value as well

        Args:
            path_ids (None or tensor): If specified, should be the specific ids corresponding to the paths to regenerate
        """
        # Call super first
        super().reset(path_ids=path_ids)

        if path_ids is None:
            path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        n_paths = len(path_ids)
        # Update radius
        self.sampled_radius[path_ids] = self.radius + \
            self.radius_noise * (-1. + 2. * torch.rand(n_paths, dtype=torch.float, device=self.device))

        # Update circle tilt
        self.sampled_circle_tilt[path_ids] = \
            self.circle_tilt_noise * (-1. + 2. * torch.rand(n_paths, dtype=torch.float, device=self.device))

        # Update axes if randomizing
        if self.randomize_axes:
            self.sampled_axes[path_ids] = torch.multinomial(
                input=0.33 * torch.ones(n_paths, 3, device=self.device),
                num_samples=3,
                replacement=False,
            )

    def _generate_pose(self, idx):
        """
        Generates the current step in the path as specified by @idx.

        The circle paths generated as in, e.g., the y-z plane, with position parameterized by the equation
        (without offsets):

            y = radius * cos(idx * 2 * pi / n_steps + pi / 2)
            z = radius * sin(idx * 2 * pi / n_steps + pi / 2)

        This generates a circular path starting at the top of the circle and rotates counterclockwise.

        Args:
            idx (tensor): Indexes for each parameterized path to grab location

        Returns:
            2-tuple:
                - tensor: (x,y,z) cartesian position of the location
                - tensor: (x,y,z,w) quaternion of the location
        """
        # Make sure idx lies in valid range
        idx = idx % self.n_steps

        path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        # Generate pose
        pos, quat = torch.zeros_like(self.sampled_center_pos), torch.zeros_like(self.sampled_center_quat)

        # Fill pos values
        pos[path_ids, self.sampled_axes[path_ids, 0]] = self.sampled_radius * torch.cos(idx * 2.0 * _PI / self.n_steps + _PI / 2.0)
        pos[path_ids, self.sampled_axes[path_ids, 1]] = self.sampled_radius * torch.sin(idx * 2.0 * _PI / self.n_steps + _PI / 2.0)

        # Fill w quat values
        quat[:, 3] = 1.0

        # Return generated values
        return pos, quat

    def _postprocess_pos(self, pos):
        """
        Additionally postprocesses @pos by rotating the circle

        Args:
            pos (tensor): (n_paths, 4) tensor where final dim is (x,y,z) cartesian position tensor to
                offset by self.center_pos

        Returns:
            tensor: offset pos tensor
        """
        path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        magnitude = pos[path_ids, self.sampled_axes[path_ids, 1]]

        pos[path_ids, self.sampled_axes[path_ids, 1]] = magnitude * torch.cos(self.sampled_circle_tilt)
        pos[path_ids, self.sampled_axes[path_ids, 2]] = magnitude * torch.sin(self.sampled_circle_tilt)

        return super()._postprocess_pos(pos=pos)


class SquarePath(TrajectoryPath):
    """
    A path consisting of a single square

    Args:
        half_size (float): half length of the generated square
        half_size_noise (float): Amount of noise to vary half length by per path generated
        plane (2-tuple): Axes that form the plane upon which the square path will be generated. Options are {x, y, z}
        randomize_axes (bool): If True, will randomize the axes for the square path between resets
        center_pos (3-array): Center (x,y,z) position for the desired loop
        center_quat (4-array): Center (x,y,z,w) quaternion rotation for the desired path
        center_pos_noise (float): Amount of noise (+/- this value) to apply when generating center path position
        center_rot_noise (float): Amount of noise (+/- this value) to apply when generating center path orientation
        square_tilt_noise (float): Amount of noise (+/- this value) to apply when generating square path orientation
        n_steps (int): Number of steps to generate in the path
        n_paths (int): How many paths to generate at a single time
        device (str): Which device to send all tensor to
        kwargs (dict): Not used; dummy var to sink extraneous variables (will print warning)
    """
    def __init__(
        self,
        half_size=0.5,
        half_size_noise=0.1,
        plane=("x", "z"),
        randomize_axes=False,
        center_pos=(0, 0, 0),
        center_quat=(0, 0, 0, 1),
        center_pos_noise=0.05,
        center_rot_noise=0.5,
        square_tilt_noise=0.785,
        n_steps=100,
        n_paths=1,
        device="cuda:0",
        **kwargs,
    ):
        # Store half_size info
        self.half_size = half_size
        self.half_size_noise = half_size_noise
        self.sampled_half_size = torch.tensor([half_size] * n_paths, dtype=torch.float, device=device)

        # Store plane info
        MAPPING = {val: i for i, val in enumerate(("x", "y", "z"))}
        self.sampled_axes = torch.zeros(n_paths, 3, dtype=torch.long, device=device)
        self.sampled_axes[:, 0] = MAPPING.pop(plane[0])
        self.sampled_axes[:, 1] = MAPPING.pop(plane[1])
        self.sampled_axes[:, 2] = list(MAPPING.values())[0]
        self.randomize_axes = randomize_axes
        self.square_tilt_noise = square_tilt_noise
        self.sampled_square_tilt = torch.zeros(n_paths, dtype=torch.float, device=device)

        # Run super init first
        super().__init__(
            center_pos=center_pos,
            center_quat=center_quat,
            center_pos_noise=center_pos_noise,
            center_rot_noise=center_rot_noise,
            n_steps=n_steps,
            n_paths=n_paths,
            device=device,
            **kwargs,
        )

    def reset(self, path_ids=None):
        """
        In addition to normal resetting, update the half size value as well

        Args:
            path_ids (None or tensor): If specified, should be the specific ids corresponding to the paths to regenerate
        """
        # Call super first
        super().reset(path_ids=path_ids)

        if path_ids is None:
            path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        n_paths = len(path_ids)
        # Update half size
        self.sampled_half_size[path_ids] = self.half_size + \
            self.half_size_noise * (-1. + 2. * torch.rand(n_paths, dtype=torch.float, device=self.device))

        # Update circle tilt
        self.sampled_square_tilt[path_ids] = \
            self.square_tilt_noise * (-1. + 2. * torch.rand(n_paths, dtype=torch.float, device=self.device))

        # Update axes if randomizing
        if self.randomize_axes:
            self.sampled_axes[path_ids] = torch.multinomial(
                input=0.33 * torch.ones(n_paths, 3, device=self.device),
                num_samples=3,
                replacement=False,
            )

    def _generate_pose(self, idx):
        """
        Generates the current step in the path as specified by @idx.

        The square paths generated as in, e.g., the y-z plane, with position parameterized by the equation
        (without offsets):

            0 <= t < 1/4 T: y = half_size, z = half_size * (-1 + 2 * t)
            1/4 T <= t < 1/2 T: y = -half_size * (-1 + 2 * (4 * t / T - 1)), z = half_size
            1/2 T <= t < 3/4 T: y = -half_size, z = half_size * (-1 + 2 * (4 * t / T - 2))
            3/4 T <= t < T: y = half_size * (-1 + 2 * (4 * t / T - 3)), z = -half_size

        This generates a square path starting at a corner of the square and rotates counterclockwise.

        Args:
            idx (tensor): Indexes for each parameterized path to grab location

        Returns:
            2-tuple:
                - tensor: (x,y,z) cartesian position of the location
                - tensor: (x,y,z,w) quaternion of the location
        """
        # Make sure idx lies in valid range
        idx = idx % self.n_steps

        path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        # Generate pose
        pos, quat = torch.zeros_like(self.sampled_center_pos), torch.zeros_like(self.sampled_center_quat)

        # Fill pos values
        pos[path_ids, self.sampled_axes[path_ids, 0]] = torch.where(idx < 0.25 * self.n_steps,
            self.sampled_half_size, torch.where(idx < 0.5 * self.n_steps,
            -self.sampled_half_size * (-1.0 + 2 * (4 * idx / self.n_steps - 1.0)), torch.where(idx < 0.75 * self.n_steps,
            -self.sampled_half_size,
            self.sampled_half_size * (-1.0 + 2 * (4 * idx / self.n_steps - 3.0))
                )
            )
        )
        pos[path_ids, self.sampled_axes[path_ids, 1]] = torch.where(idx < 0.25 * self.n_steps,
            self.sampled_half_size * (-1.0 + 2 * (4 * idx / self.n_steps - 0.0)), torch.where(idx < 0.5 * self.n_steps,
            self.sampled_half_size, torch.where(idx < 0.75 * self.n_steps,
            -self.sampled_half_size * (-1.0 + 2 * (4 * idx / self.n_steps - 2.0)),
            -self.sampled_half_size
                )
            )
        )

        # Fill w quat values
        quat[:, 3] = 1.0

        # Return generated values
        return pos, quat

    def _postprocess_pos(self, pos):
        """
        Additionally postprocesses @pos by rotating the square

        Args:
            pos (tensor): (n_paths, 4) tensor where final dim is (x,y,z) cartesian position tensor to
                offset by self.center_pos

        Returns:
            tensor: offset pos tensor
        """
        path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        magnitude = pos[path_ids, self.sampled_axes[path_ids, 1]]

        pos[path_ids, self.sampled_axes[path_ids, 1]] = magnitude * torch.cos(self.sampled_square_tilt)
        pos[path_ids, self.sampled_axes[path_ids, 2]] = magnitude * torch.sin(self.sampled_square_tilt)

        return super()._postprocess_pos(pos=pos)


class StraightPath(TrajectoryPath):
    """
    A path consisting of straight lines

    Args:
        pos_range (3-array): (x,y,z) +/- values applied to @center_pos for generating paths
        rot_range (float): +/- rotation values that can be applied when generating paths
        timestep_range (int): +/- time values that can be applied when generating paths
        pause_steps (int): Number of pause steps between path generations
        pause_range (int): +/- pause values that can be applied when generating paths
        center_pos (3-array): Center (x,y,z) position for generating paths
        center_quat (4-array): Center (x,y,z,w) quaternion rotation for the desired path
        center_pos_noise (float): Amount of noise (+/- this value) to apply when generating center path position
        center_rot_noise (float): Amount of noise (+/- this value) to apply when generating center path orientation
        n_steps (int): Number of steps to generate in the path
        n_paths (int): How many paths to generate at a single time
        device (str): Which device to send all tensor to
        kwargs (dict): Not used; dummy var to sink extraneous variables (will print warning)
    """
    def __init__(
        self,
        pos_range=(0.5, 0.5, 0.5),
        rot_range=0.785,
        timestep_range=50,
        pause_steps=10,
        pause_range=5,
        center_pos=(0, 0, 0),
        center_quat=(0, 0, 0, 1),
        center_pos_noise=0.05,
        center_rot_noise=0.5,
        n_steps=100,
        n_paths=1,
        device="cuda:0",
        **kwargs,
    ):
        # Store relevant path info
        self.pos_range = torch.tensor(pos_range, device=device).unsqueeze(0)
        self.rot_range = rot_range
        self.timestep_range = timestep_range
        self.pause_steps = pause_steps
        self.pause_range = pause_range

        # Initialize other relevant variables
        self.sampled_start_pos = torch.zeros(n_paths, 3, dtype=torch.float, device=device)
        self.sampled_start_quat = torch.zeros(n_paths, 4, dtype=torch.float, device=device)
        self.sampled_end_pos = torch.zeros_like(self.sampled_start_pos)
        self.sampled_end_quat = torch.zeros_like(self.sampled_start_quat)
        self.sampled_path_time = torch.zeros(n_paths, dtype=torch.long, device=device)
        self.sampled_pause_time = torch.zeros_like(self.sampled_path_time)

        # Run super init first
        super().__init__(
            center_pos=center_pos,
            center_quat=center_quat,
            center_pos_noise=center_pos_noise,
            center_rot_noise=center_rot_noise,
            n_steps=n_steps,
            n_paths=n_paths,
            device=device,
            **kwargs,
        )

    def reset(self, path_ids=None):
        """
        In addition to normal resetting, update the half size value as well

        Args:
            path_ids (None or tensor): If specified, should be the specific ids corresponding to the paths to regenerate
        """
        # We assume this a hard reset
        self._reset(path_ids=path_ids, soft=False)

    def _reset(self, path_ids=None, soft=False):
        """
        Overloads he default reset functionality to include soft resets, where the prior end point becomes the start
        point of the current generated path

        Args:
            path_ids (None or tensor): If specified, should be the specific ids corresponding to the paths to regenerate
            soft (bool): If True, will execute a soft reset instead of hard reset
        """
        # Only call super if hard reset is occurring
        if not soft:
            # Call super first
            super().reset(path_ids=path_ids)

        if path_ids is None:
            path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        n_paths = len(path_ids)

        # Generate new straight path
        self.sampled_start_pos[path_ids] = self.sampled_end_pos[path_ids].clone() if soft else \
            self.pos_range * (-1. + 2. * torch.rand(n_paths, 3, dtype=torch.float, device=self.device))
        self.sampled_end_pos[path_ids] = \
            self.pos_range * (-1. + 2. * torch.rand(n_paths, 3, dtype=torch.float, device=self.device))
        self.sampled_start_quat[path_ids] = self.sampled_end_quat[path_ids].clone() if soft else \
            self._generate_random_quat(n=n_paths)
        self.sampled_end_quat[path_ids] = self._generate_random_quat(n=n_paths)

        # Sample new time values
        self.sampled_path_time[path_ids] = self.n_steps + \
            torch.randint(low=-self.timestep_range, high=self.timestep_range + 1, size=(n_paths,), dtype=torch.long, device=self.device)
        self.sampled_pause_time[path_ids] = self.pause_steps + \
            torch.randint(low=-self.pause_range, high=self.pause_range + 1, size=(n_paths,), dtype=torch.long, device=self.device)

        # Reset counters for these envs
        self.current_step[path_ids] = 0

    def _generate_random_quat(self, n):
        """
        Helper function to generate random quaternion values

        To achieve this, we:
            - Sample a random unit vector in the x-y plane in half circle
            - Sample a random magnitude according to self.rot_range
            - This becomes the sampled axis-angle value --> convert to quaternion

        Args:
            n (int): Number of quaternions to generate
        """
        # Generate random unit vector in xy plane and scale randomly
        magnitude = self.rot_range * (-1. + 2. * torch.rand(n, dtype=torch.float, device=self.device))
        angle = _PI * torch.rand(n, dtype=torch.float, device=self.device)
        samples = torch.zeros(n, 3, dtype=torch.float, device=self.device)
        samples[:, 1] = magnitude * torch.cos(angle)
        samples[:, 2] = magnitude * torch.sin(angle)

        # Convert to quaternion
        samples = axisangle2quat(samples)

        # Return values
        return samples

    def _generate_pose(self, idx):
        """
        Generates the current step in the path as specified by @idx.

        First, we need to check: current_step >= sampled_path_time + sampled_pause_time --> reset path

        Then, there are two cases:
            1. current_step >= sampled_path_time --> return end point (paused at this pose)
            2. current_step < sampled_path_time --> return interpolated point

        Args:
            idx (tensor): Indexes for each parameterized path to grab location

        Returns:
            2-tuple:
                - tensor: (x,y,z) cartesian position of the location
                - tensor: (x,y,z,w) quaternion of the location
        """
        path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        # Get idx corresponding to whether current_step >= sampled_path_time + sampled_pause_time
        reset_ids = torch.nonzero(idx >= self.sampled_path_time + self.sampled_pause_time, as_tuple=True)[0]
        if len(reset_ids) > 0:
            self._reset(path_ids=reset_ids, soft=True)

        # Now generate pos and quat based on current state
        frac = (idx / self.sampled_path_time).unsqueeze(-1)
        pause_path = (idx >= self.sampled_path_time).unsqueeze(-1)
        pos = torch.where(
            pause_path,
            self.sampled_end_pos,
            self.sampled_start_pos + (self.sampled_end_pos - self.sampled_start_pos) * frac,
        )

        quat = torch.where(
            pause_path,
            self.sampled_end_quat,
            quat_slerp(quat0=self.sampled_start_quat, quat1=self.sampled_end_quat, frac=frac),
        )

        # Return generated values
        return pos, quat


class CustomPath(TrajectoryPath):
    """
    A path consisting of a custom path, as specified by a single line drawing in an SVG file. Assumes drawing should
    be read FACING the robot (robot will draw the path INVERTED horizontally)

    Args:
        fpath (str): Absolute or relative path to file to generate for this custom path. If relative path is given,
            it is assumed to be taken relative to the assets folder for this repo
        size (str): +/- (x,y) values to scale generated path to
        size_noise (2-array): Amount of xy noise to vary trajectory by per path generated
        plane (2-tuple): Axes that form the plane upon which the circle path will be generated. Options are {x, y, z}
        randomize_axes (bool): If True, will randomize the axes for the circle path between resets
        center_pos (3-array): Center (x,y,z) position for the desired loop
        center_quat (4-array): Center (x,y,z,w) quaternion rotation for the desired path
        center_pos_noise (float): Amount of noise (+/- this value) to apply when generating center path position
        center_rot_noise (float): Amount of noise (+/- this value) to apply when generating center path orientation
        tilt_noise (float): Amount of noise (+/- this value) to apply when generating path orientation
        n_steps (int): Number of steps to generate in the path
        n_paths (int): How many paths to generate at a single time
        device (str): Which device to send all tensor to
        kwargs (dict): Not used; dummy var to sink extraneous variables (will print warning)
    """
    def __init__(
        self,
        fpath,
        size=(1.0, 1.0),
        size_noise=(0.05, 0.05),
        plane=("x", "z"),
        randomize_axes=False,
        center_pos=(0, 0, 0),
        center_quat=(0, 0, 0, 1),
        center_pos_noise=0.05,
        center_rot_noise=0.5,
        tilt_noise=0.785,
        n_steps=100,
        n_paths=1,
        device="cuda:0",
        **kwargs,
    ):
        # Store info
        self.size = size
        self.size_noise = size_noise
        self.sampled_size = torch.tensor([list(size)] * n_paths, dtype=torch.float, device=device)

        # Store plane info
        MAPPING = {val: i for i, val in enumerate(("x", "y", "z"))}
        self.sampled_axes = torch.zeros(n_paths, 3, dtype=torch.long, device=device)
        self.sampled_axes[:, 0] = MAPPING.pop(plane[0])
        self.sampled_axes[:, 1] = MAPPING.pop(plane[1])
        self.sampled_axes[:, 2] = list(MAPPING.values())[0]
        self.randomize_axes = randomize_axes
        self.tilt_noise = tilt_noise
        self.sampled_tilt = torch.zeros(n_paths, dtype=torch.float, device=device)

        # Store device early so load_path method can access it
        self.device = device

        # Load path and store related info
        self.points = self._load_path(fpath=fpath)
        self.n_points = len(self.points)
        self.ratio = self.n_points / n_steps

        # Run super init first
        super().__init__(
            center_pos=center_pos,
            center_quat=center_quat,
            center_pos_noise=center_pos_noise,
            center_rot_noise=center_rot_noise,
            n_steps=n_steps,
            n_paths=n_paths,
            device=device,
            **kwargs,
        )

    def _load_path(self, fpath):
        """
        Loads a custom 2D-line path from an SVG file

        Args:
            fpath (str): Absolute or relative path to file to generate for this custom path. If relative path is given,
                it is assumed to be taken relative to the assets folder for this repo

        Returns:
            tensor: (2, N) array of points corresponding to this path, scaled and normalized such the points are
                within the unit box and centered around the origin
        """
        # Make sure file path is absolute
        if fpath[0] != "/":
            fpath = os.path.join(ASSETS_ROOT, fpath)
        # Parse the file as an element tree (assumed to be 0th element)
        root = ET.parse(fpath).getroot()
        path_elements = root.findall('.//{http://www.w3.org/2000/svg}path')
        path = [parse_path(ele.attrib['d']) for ele in path_elements][0]

        # Grab smoothed path and record bbox
        bbox = path.get_extents()
        points = path.to_polygons(closed_only=False)[0]

        # Re-scale and recenter values
        xrange_half = (bbox.x1 - bbox.x0)
        yrange_half = (bbox.y1 - bbox.y0)
        points[:, 0] = points[:, 0] / xrange_half - 1.0
        points[:, 1] = -1.0 * (points[:, 1] / yrange_half - 1.0)        # y axis is downwards

        # Return points as tensor
        return torch.tensor(points, dtype=torch.float, device=self.device)

    def reset(self, path_ids=None):
        """
        In addition to normal resetting, update the size value as well

        Args:
            path_ids (None or tensor): If specified, should be the specific ids corresponding to the paths to regenerate
        """
        # Call super first
        super().reset(path_ids=path_ids)

        if path_ids is None:
            path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        n_paths = len(path_ids)
        # Update size
        for i, (size, size_noise) in enumerate(zip(self.size, self.size_noise)):
            self.sampled_size[path_ids, i] = size + \
                size_noise * (-1. + 2. * torch.rand(n_paths, dtype=torch.float, device=self.device))

        # Update tilt
        self.sampled_tilt[path_ids] = \
            self.tilt_noise * (-1. + 2. * torch.rand(n_paths, dtype=torch.float, device=self.device))

        # Update axes if randomizing
        if self.randomize_axes:
            self.sampled_axes[path_ids] = torch.multinomial(
                input=0.33 * torch.ones(n_paths, 3, device=self.device),
                num_samples=3,
                replacement=False,
            )

    def _generate_pose(self, idx):
        """
        Generates the current step in the path as specified by @idx.

        We achieve this by using the parameterized generated path, and scaling the temporal value according to the
        ratio between n_steps and n_points. Non-integer values result in interpolation between points.

        Args:
            idx (tensor): Indexes for each parameterized path to grab location

        Returns:
            2-tuple:
                - tensor: (x,y,z) cartesian position of the location
                - tensor: (x,y,z,w) quaternion of the location
        """
        # Make sure idx lies in valid range
        idx = idx % self.n_steps

        path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        # Generate pose
        pos, quat = torch.zeros_like(self.sampled_center_pos), torch.zeros_like(self.sampled_center_quat)

        # Grab current place in path in point space
        point_idx = torch.clip(idx * self.ratio, max=self.n_points - 1)
        frac = (point_idx % 1.0).unsqueeze(-1)
        next_point_idx = torch.clip(point_idx + 1, max=self.n_points - 1)
        points = self.points[point_idx.long()] + frac * (self.points[next_point_idx.long()] - self.points[point_idx.long()])

        # Fill pos values
        for i in range(2):
            pos[path_ids, self.sampled_axes[path_ids, i]] = self.sampled_size[:, i] * points[:, i]

        # Fill w quat values
        quat[:, 3] = 1.0

        # Return generated values
        return pos, quat

    def _postprocess_pos(self, pos):
        """
        Additionally postprocesses @pos by rotating the circle

        Args:
            pos (tensor): (n_paths, 4) tensor where final dim is (x,y,z) cartesian position tensor to
                offset by self.center_pos

        Returns:
            tensor: offset pos tensor
        """
        path_ids = torch.arange(start=0, end=self.n_paths, device=self.device, dtype=torch.long)

        magnitude = pos[path_ids, self.sampled_axes[path_ids, 1]]

        pos[path_ids, self.sampled_axes[path_ids, 1]] = magnitude * torch.cos(self.sampled_tilt)
        pos[path_ids, self.sampled_axes[path_ids, 2]] = magnitude * torch.sin(self.sampled_tilt)

        return super()._postprocess_pos(pos=pos)


PATH_MAPPING = {
    "circle": CirclePath,
    "square": SquarePath,
    "straight": StraightPath,
    "custom": CustomPath,
}
