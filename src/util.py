""" utility functions """

import os
import torch


def merge_models(model, models_to_merge):
    """ load a set of T5 models with paths given in models_to_merge, perform a
        uniform weighted average of their parameters, and then load the
        parameters into the specified model object """

    # loop through models, averaging their parameters
    weight = 1. / len(models_to_merge)
    merged_state_dict = {name : 0. for name, param in model.named_parameters()}
    merged_state_dict.update({'encoder.embed_tokens.weight' : 0.,
                              'decoder.embed_tokens.weight' : 0.,
                              'lm_head.weight' : 0.})
    for path in models_to_merge:
        if path.endswith('pytorch_model.bin'):
            path = os.path.dirname(path)
        if not os.path.exists(os.path.join(path, 'pytorch_model.bin')):
            raise ValueError('paths to models must contain saved model')
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'),
                                map_location='cpu')
        for name, param in state_dict.items():
            merged_state_dict[name] += weight * param

    # load merged parameters into model
    model.load_state_dict(merged_state_dict)

