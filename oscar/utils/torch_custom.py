# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Set of custom torch classes
"""
import torch


class NonStrictModuleList(torch.nn.ModuleList):
    """
    Custom class to prevent network mismatches from breaking the code at runtime

    Use this sparingly! User should know why it's "okay" to not have specific models match between
    loading instances (e.g.: transfer learning applications where you want to learn a residual that didn't
    exist in the model initially)
    """

    def load_state_dict(self, state_dict, strict=True):
        # Override super implementation so that code doesn't break if we're loading a slightly mismatched model
        # e.g.: If residual does/n't exist in original loaded model but does/n't in this one
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
        # Raise a warning if we get some unexpected keys
        if len(missing_keys) > 0:
            print(f"Got missing keys when loading {self._get_name()}:\n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Got unexpected keys when loading {self._get_name()}:\n{unexpected_keys}")

        # Return keys as normal
        return missing_keys, unexpected_keys

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Override super implementation so that code doesn't break if we're loading a slightly mismatched model
        # e.g.: If residual does/n't exist in original loaded model but does/n't in this one
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)
        # Raise a warning if we get some unexpected keys
        if len(missing_keys) > 0:
            print(f"Got missing keys when loading {self._get_name()}:\n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Got unexpected keys when loading {self._get_name()}:\n{unexpected_keys}")


class NonStrictModule(torch.nn.Module):
    """
    Custom class to prevent network mismatches from breaking the code at runtime

    Use this sparingly! User should know why it's "okay" to not have specific models match between
    loading instances (e.g.: transfer learning applications where you want to learn a residual that didn't
    exist in the model initially)
    """

    def load_state_dict(self, state_dict, strict=True):
        # Override super implementation so that code doesn't break if we're loading a slightly mismatched model
        # e.g.: If residual does/n't exist in original loaded model but does/n't in this one
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)
        # Raise a warning if we get some unexpected keys
        if len(missing_keys) > 0:
            print(f"Got missing keys when loading {self._get_name()}:\n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Got unexpected keys when loading {self._get_name()}:\n{unexpected_keys}")

        # Return keys as normal
        return missing_keys, unexpected_keys

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Override super implementation so that code doesn't break if we're loading a slightly mismatched model
        # e.g.: If residual does/n't exist in original loaded model but does/n't in this one
        # metadata = getattr(state_dict, '_metadata', None)
        # local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)
        # Raise a warning if we get some unexpected keys
        # input(f"missing keys:\n{missing_keys}")
        # input(f"unexpected keys:\n{unexpected_keys}")
        # input(f"err msg:\n{error_msgs}")
        if len(missing_keys) > 0:
            print(f"Got missing keys when loading {self._get_name()}:\n{missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Got unexpected keys when loading {self._get_name()}:\n{unexpected_keys}")


class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def maybe_no_grad(no_grad):
    return torch.no_grad() if no_grad else dummy_context_mgr()
