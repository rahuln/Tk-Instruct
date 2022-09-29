""" script to evaluate the weighted average of a given set of experts on a
    given task category over a grid of different averaging weights """

from argparse import ArgumentParser
from glob import glob
from itertools import product
import json
import os
import subprocess
import sys

import numpy as np


# command-line arguments
parser = ArgumentParser()
parser.add_argument('cfg_file', type=str, help='path to config file')
parser.add_argument('--models', nargs='+',
                    help='paths to models to load and merge')
parser.add_argument('--category', type=str, default=None,
                    help='category to evaluate on (directory in data_dir)')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='name of experiment (e.g., number of training steps)')
parser.add_argument('--data_dir', type=str, default='data/splits/default',
                    help='data directory for evaluation tasks')
parser.add_argument('--eval_dirname', type=str, default='test',
                    help='name for evaluation results directory')
parser.add_argument('--max_num_instances_per_task', type=int, default=None,
                    help='maximum number of training instances per task')
parser.add_argument('--index', type=int, default=None,
                    help='index of Slurm array job')
args = parser.parse_args()

# load config file, select merging weights using index from command line
with open(args.cfg_file, 'r') as f:
    cfg = json.load(f)
weights_grid = list(product(*[np.arange(0., 1.1, 0.1)
                              for _ in range(len(args.models))]))
weights_grid = [elem for elem in weights_grid if np.isclose(np.sum(elem), 1)]
weights = weights_grid[args.index]
dataset = cfg.get('dataset', 'niv2')
use_dev = cfg.get('use_dev', False)

# construct strings for paths to models to merge and merging weights
models_to_merge = ','.join(args.models)
merging_weights = ','.join([f'{w:.1f}' for w in weights])
merging_weights_str = f'weights-{merging_weights.replace(",", "-")}'

# create output directory
output_dir = os.path.join('results', dataset, 'tk-instruct-base-merged',
                          'evaluate', args.eval_dirname, args.exp_name,
                          args.category, merging_weights_str)
os.makedirs(output_dir, exist_ok=True)

# check for existing results
if os.path.exists(os.path.join(output_dir, 'metrics.json')):
    print('results already exist, exiting...')
    sys.exit()

# set up command
cmd = ['python', 'src/run_s2s.py',
       '--do_predict',
       '--predict_with_generate',
       '--evaluation_strategy=no',
       '--model_name_or_path=allenai/tk-instruct-base-def-pos',
       f'--models_to_merge={models_to_merge}',
       f'--merging_weights={merging_weights}',
       '--max_source_length=1024',
       '--max_target_length=128',
       '--generation_max_length=128',
       '--max_num_instances_per_eval_task=100',
       '--add_task_name=False',
       '--add_task_definition=True',
       '--num_pos_examples=2',
       '--num_neg_examples=0',
       '--add_explanation=False',
       '--tk_instruct=False',
       f'--data_dir={args.data_dir}/{args.category}',
       '--task_dir=data/tasks',
       f'--output_dir={output_dir}',
       '--overwrite_output_dir',
       '--cache_dir=/gscratch/ark/rahuln/.cache',
       '--overwrite_cache',
       '--per_device_eval_batch_size=4']

# specify max_num_instances_per_task
if args.max_num_instances_per_task is not None:
    max_num_instances = args.max_num_instances_per_task
    cmd.append(f'--max_num_instances_per_task={max_num_instances}')

# use dev/test split of test set if specified
if use_dev:
    cmd.extend(['--do_eval', '--use_dev'])

# print command to log file
print(' '.join(cmd))
sys.stdout.flush()
sys.stderr.flush()
subprocess.call(cmd)

# write command out to file
with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
    cmd_str = ' '.join(cmd).replace(' --', ' \\\n    --')
    f.write(cmd_str)

