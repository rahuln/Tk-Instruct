""" script to compute and save embeddings of task instances/inputs for
    NaturalInstructions tasks """

from argparse import ArgumentParser
from glob import glob
import json
import os
import string

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


parser = ArgumentParser(description='script to compute and save embeddings '
                                    'of task instances/inputs for '
                                    'NaturalInstructions tasks')
parser.add_argument('--task_dir', type=str, default='data/tasks',
                    help='path to tasks directory')
parser.add_argument('--tasks_file', type=str, default=None,
                    help='list of tasks to include, excluding all others')
parser.add_argument('--model_name_or_path', type=str, default='roberta-base',
                    help='model to use for computing embeddings')
parser.add_argument('--use_encoder', action='store_true',
                    help='use only the encoder of an encoder-decoder model')
parser.add_argument('--start_index', type=int, default=50,
                    help='start index of instances to include (inclusive)')
parser.add_argument('--end_index', type=int, default=100,
                    help='end index of instances to include (exclusive)')
parser.add_argument('--max_length', type=int, default=512,
                    help='maximum number of input tokens per instance')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size to use when iterating over instances')
parser.add_argument('--pooling', type=str, default='mean',
                    choices=['mean', 'cls'],
                    help='type of pooling to perform over output embeddings')
parser.add_argument('--normalize', action='store_true',
                    help='normalize embeddings to have unit L2 norm')
parser.add_argument('--mean_over_task', action='store_true',
                    help='store mean embeddings across instances within each '
                         'task rather than per-instance embeddings for each '
                         'task')
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

    # get list of tasks to restrict to, if specified
    tasks_to_include = None
    if args.tasks_file is not None:
        with open(args.tasks_file, 'r') as f:
            tasks_to_include = set([elem.strip() for elem in f.readlines()])

    # get task names and descriptions
    task_files = sorted(glob(os.path.join(args.task_dir, '*.json')))
    task_names = list()
    task_categories = list()
    task_instances = dict()
    task_instance_ids = dict()
    for fname in tqdm(task_files, desc='loading task info'):
        with open(fname, 'r') as f:
            task_info = json.load(f)
        task_name = os.path.basename(fname).replace('.json', '')
        if tasks_to_include is not None and task_name not in tasks_to_include:
            continue
        task_names.append(task_name)
        task_category = task_info['Categories'][0].lower().replace(' ', '_')
        task_categories.append(task_category)
        instances = [elem['input'] for elem in task_info['Instances']]
        task_instances[task_name] = instances[args.start_index:args.end_index]
        instance_ids = [elem['id'] for elem in task_info['Instances']]
        task_instance_ids[task_name] = instance_ids[args.start_index:args.end_index]

    # load model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      cache_dir=args.cache_dir)
    model = model.cuda()
    if args.use_encoder and hasattr(model, 'encoder'):
        model = model.encoder

    # compute embeddings of instances for each task
    embeddings = dict()
    lengths = list()
    for i, task_name in enumerate(tqdm(task_names,
                                       desc='computing embeddings')):

        # initialize dict to store embeddings, data loader for instances
        all_text = task_instances[task_name]
        task_embeddings = list()
        loader = DataLoader(torch.arange(len(all_text)),
                            batch_size=args.batch_size)

        # iterate through batches of instance and compute embeddings
        for idxs in loader:
            text = [all_text[idx] for idx in idxs]
            lengths.extend(list(map(len, list(map(tokenizer.tokenize, text)))))
            inputs = tokenizer(text, padding=True, truncation=True,
                               max_length=args.max_length,
                               return_tensors='pt').to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)

            # apply specific pooling type
            if args.pooling == 'mean':
                embedding = mean_pooling(outputs.last_hidden_state,
                                         inputs.attention_mask)
            elif args.pooling == 'cls':
                embedding = outputs.last_hidden_state[:, 0, :]

            task_embeddings.append(embedding.detach().cpu())

        embeddings[task_name] = torch.cat(task_embeddings)

    # normalize embeddings
    if args.normalize and not args.mean_over_task:
        for task_name, emb in embeddings.items():
            embeddings[task_name] = F.normalize(emb, p=2, dim=-1)

    # save to file
    if args.savename is not None:

        # if specified, average embeddings over all instances within a task,
        # otherwise save embeddings for individual instance for each task
        if args.mean_over_task:
            mean_emb = list()
            for task_name in task_names:
                mean_emb.append(torch.mean(embeddings[task_name], dim=0))
            mean_emb = torch.stack(mean_emb)
            if args.normalize:
                mean_emb = F.normalize(mean_emb, p=2, dim=-1)

            results = {
                'task_names' : task_names,
                'task_categories' : task_categories,
                'task_instance_ids' : task_instance_ids,
                'embeddings' : mean_emb
            }
        else:
            results = {
                'task_names' : task_names,
                'task_categories' : task_categories,
                'task_instance_ids' : task_instance_ids,
                'embeddings' : embeddings
            }

        # save embeddings to file
        model_name = args.model_name_or_path.replace('/', '-')
        category_str = '-task' if args.mean_over_task else ''
        savename = (f'{args.savename}-{model_name}{category_str}'
                    f'-{args.pooling}.pt')
        torch.save(results, savename)

    # print statistics on instance text inputs
    print(f'mean instance length: {np.mean(lengths):.3f}')
    print(f'maximum instance length: {np.max(lengths):.3f}')
    frac_longer = np.mean(np.array(lengths) > args.max_length)
    print(f'fraction longer than tokenizer limit: {frac_longer:.3f}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
