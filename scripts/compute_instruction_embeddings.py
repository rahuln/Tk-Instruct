""" script to compute and save embeddings of task descriptions/instructions
    for NaturalInstructions tasks """

from argparse import ArgumentParser
from glob import glob
import json
import os
import string

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
parser.add_argument('--add_pos_examples', action='store_true',
                    help='encode positive examples as part of instruction')
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
    task_pos_examples = list()
    for fname in tqdm(task_files, desc='loading task info'):
        with open(fname, 'r') as f:
            task_info = json.load(f)
        task_names.append(os.path.basename(fname).replace('.json', ''))
        if isinstance(task_info['Definition'], list):
            task_descriptions.append(task_info['Definition'][0])
        else:
            task_descriptions.append(task_info['Definition'].strip())
        task_categories.append(task_info['Categories'][0])
        task_pos_examples.append(task_info['Positive Examples'][:2])

    # load model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              cache_dir=args.cache_dir)
    model = AutoModel.from_pretrained(args.model_name_or_path,
                                      cache_dir=args.cache_dir)
    model = model.cuda()

    # compute embedding for each task description
    embeddings = list()
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
                if len(tokenizer(new_text)['input_ids']) <= args.max_length:
                    pos_str_list.append(pos_example_str)
                else:
                    break
            text = text + ''.join(pos_str_list)

        inputs = tokenizer(text, max_length=args.max_length,
                           return_tensors='pt').to('cuda')
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        if args.pooling == 'mean':
            embedding = mean_pooling(outputs.last_hidden_state,
                                     inputs.attention_mask)
        elif args.pooling == 'cls':
            embedding = output.last_hidden_state[:, 0, :]
        embeddings.append(embedding.detach().cpu())
    embeddings = torch.stack(embeddings).squeeze()

    # save to file
    if args.savename is not None:
        results = {
            'task_names' : task_names,
            'task_descriptions' : task_descriptions,
            'task_categories' : task_categories,
            'embeddings' : embeddings
        }
        model_name = args.model_name_or_path.replace('/', '-')
        pos_str = '-pos' if args.add_pos_examples else ''
        savename = (f'{args.savename}-{model_name}-{args.pooling}-'
                    f'def{pos_str}.pt')
        torch.save(results, savename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
