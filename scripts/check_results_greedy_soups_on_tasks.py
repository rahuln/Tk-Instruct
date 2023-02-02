""" script to summarize results in a specified directory when evaluation is
    performed on individual tasks (i.e., per-task results sub-directories) """

from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from itertools import chain
import json
import os
from termcolor import colored

import numpy as np
from tqdm import tqdm


# command-line arguments
parser = ArgumentParser()
parser.add_argument('basedir', type=str, help='results directory')
parser.add_argument('--task_file', type=str, default=None,
                    help='list of tasks to speed up loading task info files')
args = parser.parse_args()


# mappings from task to category and category to values for each metric
metric_names = ['exact_match', 'rouge1', 'rougeL']
task_to_category = dict()
category_to_metric = {name : defaultdict(lambda: list())
                      for name in metric_names}

# get task info files
if args.task_file is not None:
    with open(args.task_file, 'r') as f:
        tasks = [line.strip() for line in f.readlines()]
    task_info_files = [f'data/tasks/{task}.json' for task in tasks]
else:
    task_info_files = sorted(glob('data/tasks/*.json'))

# get mapping from task name to category
for fname in tqdm(task_info_files, desc='getting task categories'):
    task = os.path.basename(fname).replace('.json', '')
    with open(fname, 'r') as f:
        task_info = json.load(f)
    category = task_info['Categories'][0].lower().replace(' ', '_')
    task_to_category[task] = category

# get results files
files = sorted(glob(os.path.join(args.basedir, '**', 'metrics.json'),
               recursive=True))

# collect metric values for each task, add to list for task category
for fname in tqdm(files, desc='collecting results'):
    task = os.path.basename(os.path.dirname(fname))
    with open(fname, 'r') as f:
        metrics = json.load(f)
    category = task_to_category[task]
    for name in metric_names:
        value = metrics[f'predict_{name}_for_{task}']
        category_to_metric[name][category].append(value)

# calculate mean over categories for each metric
mean_values = dict()
for name in metric_names:
    mean_values[name] = [np.mean(category_to_metric[name][key]) for key in
                         sorted(list(category_to_metric[name].keys()))]
print('number of categories:', len(mean_values['rougeL']))
print('mean over categories:')
for name, values in mean_values.items():
    print(f'  {name}: {np.mean(values):.4f}')

# calculate mean over tasks for each metric
all_values = list(chain(*list(category_to_metric['rougeL'].values())))
error_msg = colored('(does not match task file)', 'red') \
            if len(all_values) != len(tasks) else ''
print('number of tasks:', len(all_values), error_msg)
print('mean over tasks:')
for name in metric_names:
    all_values = list(chain(*list(category_to_metric[name].values())))
    print(f'  {name}: {np.mean(all_values):.4f}')

