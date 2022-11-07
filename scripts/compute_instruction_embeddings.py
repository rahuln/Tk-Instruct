""" script to compute and save embeddings of task descriptions/instructions
    for NaturalInstructions tasks """

from argparse import ArgumentParser
from glob import glob
import json
import os
import string

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


parser = ArgumentParser(description='script to compute and save embeddings '
                                    'of task descriptions/instructions for '
                                    'NaturalInstructions tasks')
parser.add_argument('--task_dir', type=str, default='data/tasks',
                    help='path to tasks directory')
parser.add_argument('--tasks_file', type=str, default=None,
                    help='list of tasks to include, excluding all others')
parser.add_argument('--model_name_or_path', type=str, default='roberta-base',
                    help='model to use for computing embeddings')
parser.add_argument('--use_encoder', action='store_true',
                    help='use only the encoder of an encoder-decoder model')
parser.add_argument('--desc_max_length', type=int, default=1024,
                    help='maximum number of input tokens per description')
parser.add_argument('--tokenizer_max_length', type=int, default=512,
                    help='maximum number of input tokens for tokenizer')
parser.add_argument('--pooling', type=str, default='mean',
                    choices=['mean', 'cls'],
                    help='type of pooling to perform over output embeddings')
parser.add_argument('--add_pos_examples', action='store_true',
                    help='encode positive examples as part of instruction')
parser.add_argument('--normalize', action='store_true',
                    help='normalize embeddings to have unit L2 norm')
parser.add_argument('--mean_over_category', action='store_true',
                    help='store mean embeddings within each task category '
                         'rather than per-task embeddings')
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
    task_descriptions = list()
    task_categories = list()
    task_pos_examples = list()
    for fname in tqdm(task_files, desc='loading task info'):
        with open(fname, 'r') as f:
            task_info = json.load(f)
        task_name = os.path.basename(fname).replace('.json', '')
        if tasks_to_include is not None and task_name not in tasks_to_include:
            continue
        task_names.append(task_name)
        if isinstance(task_info['Definition'], list):
            task_descriptions.append(task_info['Definition'][0])
        else:
            task_descriptions.append(task_info['Definition'].strip())
        task_category = task_info['Categories'][0].lower().replace(' ', '_')
        task_categories.append(task_category)
        task_pos_examples.append(task_info['Positive Examples'][:2])

    # load model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      cache_dir=args.cache_dir)
    model = model.cuda()
    if args.use_encoder and hasattr(model, 'encoder'):
        model = model.encoder

    # compute embedding for each task description
    embeddings = list()
    lengths = list()
    for i, description in enumerate(tqdm(task_descriptions,
                                         desc='computing embeddings')):
        text = f'Definition: {description}\n\n'

        # add positive examples to instruction string
        if args.add_pos_examples:
            pos_str_list = list()
            for idx, pos_example in enumerate(task_pos_examples[i]):
                pos_example_str = f' Positive Example {idx+1} -\n'
                pos_example_str += f'Input: {pos_example["input"].strip()}'
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += '.'
                pos_example_str += '\n'
                pos_example_str += f' Output: {pos_example["output"].strip()}'
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += '.'
                pos_example_str += '\n\n'
                new_text = text + ' '.join(pos_str_list) + pos_example_str
                if len(tokenizer(new_text)['input_ids']) <= args.desc_max_length:
                    pos_str_list.append(pos_example_str)
                else:
                    break
            text = text + ''.join(pos_str_list)

        lengths.append(len(tokenizer.tokenize(text)) + 2)
        inputs = tokenizer(text, truncation=True,
                           max_length=args.tokenizer_max_length,
                           return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        if args.pooling == 'mean':
            embedding = mean_pooling(outputs.last_hidden_state,
                                     inputs.attention_mask)
        elif args.pooling == 'cls':
            embedding = outputs.last_hidden_state[:, 0, :]
        embeddings.append(embedding.detach().cpu())
    embeddings = torch.stack(embeddings).squeeze()

    # normalize embeddings
    if args.normalize and not args.mean_over_category:
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    # save to file
    if args.savename is not None:

        # if specified, average embeddings over all tasks within a category,
        # otherwise save embeddings for individual tasks
        if args.mean_over_category:
            unique_categories = sorted(set(task_categories))
            category_embeddings = list()
            for category in unique_categories:
                idx = [i for i, cat in enumerate(task_categories)
                       if cat == category]
                category_embeddings.append(embeddings[idx, :].mean(dim=0))
            embeddings = torch.stack(category_embeddings).squeeze()

            if args.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            results = {
                'categories' : unique_categories,
                'embeddings' : embeddings,
            }
        else:
            results = {
                'task_names' : task_names,
                'task_descriptions' : task_descriptions,
                'task_categories' : task_categories,
                'embeddings' : embeddings
            }

        model_name = args.model_name_or_path.replace('/', '-')
        pos_str = '-pos' if args.add_pos_examples else ''
        category_str = '-category' if args.mean_over_category else ''
        savename = (f'{args.savename}-{model_name}{category_str}'
                    f'-{args.pooling}-def{pos_str}.pt')
        torch.save(results, savename)

    print(f'mean instruction length: {np.mean(lengths):.3f}')
    print(f'maximum instruction length: {np.max(lengths):.3f}')
    frac_longer = np.mean(np.array(lengths) > 512)
    print(f'fraction longer than tokenizer limit: {frac_longer:.3f}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
