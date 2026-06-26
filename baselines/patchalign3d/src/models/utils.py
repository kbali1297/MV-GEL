#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import logging
from collections import defaultdict, namedtuple

import torch
import torch.nn as nn

from termcolor import colored


def get_missing_parameters_message(keys):
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join("  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items())
    return msg


def get_unexpected_parameters_message(keys):
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join("  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items())
    return msg


def _strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return
    for key in keys:
        newkey = key[len(prefix):]
        state_dict[newkey] = state_dict.pop(key)
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        for key in list(metadata.keys()):
            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys):
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group):
    if len(group) == 0:
        return ""
    if len(group) == 1:
        return "." + group[0]
    return ".{" + ", ".join(group) + "}"


def _named_modules_with_dup(model, prefix=""):
    yield prefix, model
    for name, module in model._modules.items():
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)


_IncompatibleKeys = namedtuple("_IncompatibleKeys", ["missing_keys", "unexpected_keys"])


def load_state_dict(module, state_dict, strict=True):
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None

    if strict:
        missing_keys = [key for key in missing_keys if not key.endswith("num_batches_tracked")]
    if len(missing_keys) > 0:
        error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join('"' + k + '"' for k in missing_keys)))
    if len(unexpected_keys) > 0:
        error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join('"' + k + '"' for k in unexpected_keys)))
    if len(error_msgs) > 0:
        raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(module.__class__.__name__, "\n\t".join(error_msgs)))
    return _IncompatibleKeys(missing_keys, unexpected_keys)


def _log_api_usage(identifier: str):
    logger = logging.getLogger("torchvision")
    if not logger.handlers:
        return
    from torch._C._log_api_usage_once import log_api_usage_once
    log_api_usage_once(identifier)
