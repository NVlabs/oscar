# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Collection of utilities for dealing with observations
"""
from collections import OrderedDict
import torch
import numpy as np


class DictConverter:
    """
    Simple class to help convert between a dictionary and a tensor. This class will take in a template of expected
    keywords, corresponding tensors, and dimension along which to concatenate, and can convert them into a concatenated,
    single 2D (B, D) tensor (useful for serializing / deserializing when using rollout storage).

    Note: Assumes the batch size = first dimension = consistent across all entries in the dict!

    Args:
        dict_template (dict): Dummy dictionary containing expected key values and corresponding tensor shapes
    """
    def __init__(self, dict_template):
        # Extract useful information from dictionary
        self._shape_info = OrderedDict()
        self._first_key = None
        for key, tensor in dict_template.items():
            # Store first key (just so we can index batch information later)
            if self._first_key is None:
                self._first_key = key
            # Create corresponding entry in internal var
            self._shape_info[key] = tensor.shape[1:]

        # Compute flattened shape
        self.flattened_dim = self.to_tensor(dict_template).shape[-1]

    def to_tensor(self, input_dict):
        """
        Converts a dictionary to a concatenated tensor

        Args:
            input_dict (dict): Dictionary to convert to tensor

        Returns:
            Tensor: Concatenated tensor
        """
        # Loop over all keys and create the tensor
        B = input_dict[self._first_key].shape[0]
        return torch.cat([input_dict[k].reshape(B, -1) for k in self._shape_info.keys()], dim=-1)

    def to_dict(self, input_tensor):
        """
        Parses a single tensor into keyword-mapped components

        Args:
            input_tensor (Tensor): Tensor to convert to dictionary

        Returns:
            OrderedDict: Keyword-mapped tensor components based on internal template
        """
        # Loop over all keys and construct dictionary
        out_dict = OrderedDict()
        idx = 0
        for key, shape in self._shape_info.items():
            out_dict[key] = input_tensor.narrow(-1, idx, np.prod(shape)).reshape(-1, *shape)
            idx += np.prod(shape)

        return out_dict
