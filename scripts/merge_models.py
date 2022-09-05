""" script to merge T5 models by averaging their parameters and save the result
    to a specified directory """

from argparse import ArgumentParser
import os

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# command-line arguments
parser = ArgumentParser(description='script to merge T5 models by averaging '
                                    'their parameters and save the result to '
                                    'a specified directory')
parser.add_argument('models', nargs='+',
                    help='list of paths to models to merge')
parser.add_argument('--base-model', type=str,
                    default='allenai/tk-instruct-base-def-pos',
                    help='name/path of base model')
parser.add_argument('--cache-dir', type=str,
                    default='/gscratch/ark/rahuln/.cache',
                    help='path to Huggingface cache')
parser.add_argument('--outdir', type=str, default='merged-model',
                    help='directory to save merged model and tokenizer')


if __name__ == '__main__':
    args = parser.parse_args()

    # check for saved results
    if os.path.exists(args.outdir) and len(os.listdir(args.outdir)) > 0:
        raise ValueError('directory already exists and is not empty')
    os.makedirs(args.outdir, exist_ok=True)

    # load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.base_model,
                                            cache_dir=args.cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.base_model,
                                                       cache_dir=args.cache_dir)

    # loop through models, averaging their parameters
    weight = 1. / len(args.models)
    merged_state_dict = {name : 0. for name, param in model.named_parameters()}
    merged_state_dict.update({'encoder.embed_tokens.weight' : 0.,
                              'decoder.embed_tokens.weight' : 0.,
                              'lm_head.weight' : 0.})
    for path in args.models:
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

    # save tokenizer and model to output directory
    tokenizer.save_pretrained(args.outdir)
    model.save_pretrained(args.outdir)

