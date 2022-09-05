""" compare performance of instruction-tuned experts to a base model """

from argparse import ArgumentParser
import os
import json
from glob import glob
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


parser = ArgumentParser(description='compare performance of instruction-tuned '
                                    'experts to a base model')
parser.add_argument('--base-model-dir', type=str,
                    default='results/natural-instructions-v2/tk-instruct-base/'
                            'evaluate-on-train',
                    help='path to metric file for base model')
parser.add_argument('--expert-model-dir', type=str,
                    default='results/natural-instructions-v2/'
                            'tk-instruct-base-experts/evaluate-on-train/'
                            'category',
                    help='path to enclosing directory of all expert results')
parser.add_argument('--correlation', action='store_true',
                    help='calculate and print correlations between task '
                         'category size and performance difference')
parser.add_argument('--savename', type=str, default=None,
                    help='path to save matrix of performance differences to')
args = parser.parse_args()


def mean_offdiag(X, axis=None):
    X = X.copy()
    X[np.diag_indices_from(X)] = np.nan
    return np.nanmean(X, axis=axis)

with open(os.path.join(args.base_model_dir, 'metrics.json'), 'r') as f:
    base_metrics = json.load(f)

fn = lambda x: 'rougeL_for' in x and 'for_task' not in x
base_metrics = {key : value for key, value in base_metrics.items() if fn(key)}
tasks = sorted([key.replace('predict_rougeL_for_', '') for key in base_metrics.keys()])
base_values = np.array([base_metrics[f'predict_rougeL_for_{task}'] for task in tasks])

files = sorted(glob(os.path.join(args.expert_model_dir, '**', 'metrics.json'),
                    recursive=True))
values = np.zeros((len(files), len(tasks)))

expert_tasks = list()
for i, fname in enumerate(files):
    expert_task = os.path.basename(os.path.dirname(fname))
    expert_tasks.append(expert_task)
    with open(fname, 'r') as f:
        metrics = json.load(f)
    expert_values = np.array([metrics[f'predict_rougeL_for_{task}'] for task in tasks])
    values[i] = expert_values - base_values

print(tasks)
print(expert_tasks)
sys.exit()

if args.savename is not None:
    np.save(args.savename, values)

if args.correlation:

    fmt = 'data/splits/category/{category}/train_tasks.txt'
    num_tasks = list()
    for task in tasks:
        fname = fmt.format(category=task)
        with open(fname, 'r') as f:
            num_tasks.append(len(f.readlines()))

    print('diag, pearson:', pearsonr(num_tasks, np.diag(values)))
    print('diag, spearman:', spearmanr(num_tasks, np.diag(values)))
    print('mean off-diag, pearson:', pearsonr(num_tasks, mean_offdiag(values, axis=1)))
    print('mean off-diag, spearman:', spearmanr(num_tasks, mean_offdiag(values, axis=1)))

