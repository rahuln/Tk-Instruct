""" script to compute and save embeddings of task descriptions/instructions
    for NaturalInstructions tasks """

from argparse import ArgumentParser
from glob import glob
import json
import os

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


parser = ArgumentParser(description='script to compute and save embeddings '
                                    'of task descriptions/instructions for '
                                    'NaturalInstructions tasks')
parser.add_argument('--task_dir', type=str, default='data/tasks',
                    help='path to tasks directory')
parser.add_argument('--model_name_or_path', type=str, default='roberta-base',
                    help='model to use for computing embeddings')
parser.add_argument('--max_length', type=int, default=512,
                    help='maximum number of input tokens per description')
parser.add_argument('--pooling', type=str, default='mean',
                    choices=['mean', 'cls'],
                    help='type of pooling to perform over output embeddings')
parser.add_argument('--savename', type=str, default=None,
                    help='path to for saving output embeddings')
parser.add_argument('--cache_dir', default='/gscratch/ark/rahuln/.cache',
                    help='Huggingface cache directory')


def mean_pooling(token_embeddings, attention_mask):
    """ mean pooling over token embeddings with given attention mask """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main(args):
    """ main function """

    # get task names and descriptions
    task_files = sorted(glob(os.path.join(args.task_dir, '*.json')))
    task_names = list()
    task_descriptions = list()
    task_categories = list()
    for fname in tqdm(task_files, desc='loading task info'):
        with open(fname, 'r') as f:
            task_info = json.load(f)
        task_names.append(os.path.basename(fname).replace('.json', ''))
        task_descriptions.append(task_info['Definition'][0])
        task_categories.append(task_info['Categories'][0])

    # load model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      cache_dir=args.cache_dir)
    model = model.cuda()

    # compute embedding for each task description
    embeddings = list()
    for description in tqdm(task_descriptions, desc='computing embeddings'):
        inputs = tokenizer(description, max_length=args.max_length,
                           return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        if args.pooling == 'mean':
            embedding = mean_pooling(outputs.last_hidden_state,
                                     inputs.attention_mask)
        elif args.pooling == 'cls':
            embedding = output.last_hidden_state[:, 0, :]
        embeddings.append(embedding.detach().cpu())
    embeddings = torch.stack(embeddings)

    # save to file
    if args.savename is not None:
        results = {
            'task_names' : task_names,
            'task_descriptions' : task_descriptions,
            'task_categories' : task_categories,
            'embeddings' : embeddings
        }
        model_name = args.model_name_or_path.replace('/', '-')
        savename = f'{args.savename}-{model_name}-{args.pooling}.pt'
        torch.save(results, savename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
