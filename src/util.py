""" utility functions """

import os
import re

import torch


def send_to_device(state_dict, device):
    """ send all parameters in state_dict to a given device """
    for name, param in state_dict.items():
        state_dict[name] = param.to(device)
    return state_dict


def merge_models(model, models_to_merge, weights=None, pattern='.*'):
    """ load a set of T5 models with paths given in models_to_merge, perform a
        uniform weighted average of their parameters, and then load the
        parameters into the specified model object """

    # set weights, check that number of weights and models is the same
    if weights is None:
        weights = [1. / len(models_to_merge)] * len(models_to_merge)
    if len(weights) != len(models_to_merge):
        raise ValueError('number of models and weights must match')

    # loop through models, averaging their parameters
    model_state_dict = model.state_dict()
    merged_state_dict = {name : 0. for name, param in model.named_parameters()}
    merged_state_dict.update({'encoder.embed_tokens.weight' : 0.,
                              'decoder.embed_tokens.weight' : 0.,
                              'lm_head.weight' : 0.})
    for path, weight in zip(models_to_merge, weights):
        if isinstance(path, str):
            if path.endswith('pytorch_model.bin'):
                path = os.path.dirname(path)
            if not os.path.exists(os.path.join(path, 'pytorch_model.bin')):
                raise ValueError('paths to models must contain saved model')
            state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'),
                                    map_location='cpu')
        else:
            state_dict = path
        for name, param in state_dict.items():
            # only update parameter if its name matches the pattern
            if re.match(pattern, name):
                merged_state_dict[name] += weight * param
            else:
                merged_state_dict[name] = model_state_dict[name]

    # load merged parameters into model
    model.load_state_dict(merged_state_dict)


def merge_state_dicts(state_dict1, state_dict2, num_averaged=1, pattern='.*'):
    """ merge parameters for the given state_dicts by averaging, where the
        first state_dict is already an average over num_averaged models """

    # ensure that all parameters are on the CPU
    state_dict1 = send_to_device(state_dict1, 'cpu')
    state_dict2 = send_to_device(state_dict2, 'cpu')

    # iterate through state_dicts, merging parameters
    denom = num_averaged + 1
    merged_state_dict = dict()
    for name, param1 in state_dict1.items():
        # only update parameter if its name matches the pattern
        if re.match(pattern, name):
            param2 = state_dict2[name]
            merged_state_dict[name] = (num_averaged * param1 + param2) / denom
        else:
            merged_state_dict[name] = param1

    return merged_state_dict

