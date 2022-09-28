""" script to evaluate interpolation between tk-instruct-base and specialized
    expert on a given dataset """

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
parser.add_argument('--exp_name', type=str, default='exp',
                    help='name of experiment (e.g., number of training steps)')
parser.add_argument('--data_dir', type=str, default='data/splits/default',
                    help='data directory for evaluation tasks')
parser.add_argument('--eval_dirname', type=str, default='test',
                    help='name for evaluation results directory')
parser.add_argument('--num_weights', type=int, default=11,
                    help='number of interpolation weights in grid')
parser.add_argument('--max_num_instances_per_task', type=int, default=100,
                    help='maximum number of training instances per task')
parser.add_argument('--index', type=int, default=None,
                    help='index of Slurm array job')
args = parser.parse_args()

# load config file, select task category using index from command line
with open(args.cfg_file, 'r') as f:
    cfg = json.load(f)
weights = np.linspace(0, 1, args.num_weights)
grid = list(product(cfg['categories'], weights))
category, weight = grid[args.index]
dataset = cfg.get('dataset', 'niv2')

# specify model path
model_name_or_path = os.path.join('results', dataset,
                                  'tk-instruct-base-experts', 'train',
                                  args.exp_name, category)

# specify path to base model
base_model_path = os.path.join('results', 'niv2', 'tk-instruct-base',
                               'saved_model')

# construct strings for paths to models to merge and merging weights
models_to_merge = model_name_or_path + ',' + base_model_path
merging_weights = str(weight) + ',' + str(1 - weight)

# create output directory
output_dir = os.path.join('results', dataset, 'tk-instruct-base-experts',
                          'evaluate', args.eval_dirname, args.exp_name,
                          category, f'weight-{weight:.1f}')
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
       f'--max_num_instances_per_task={args.max_num_instances_per_task}',
       '--max_num_instances_per_eval_task=100',
       '--add_task_name=False',
       '--add_task_definition=True',
       '--num_pos_examples=2',
       '--num_neg_examples=0',
       '--add_explanation=False',
       '--tk_instruct=False',
       f'--data_dir={args.data_dir}',
       '--task_dir=data/tasks',
       f'--output_dir={output_dir}',
       '--overwrite_output_dir',
       '--cache_dir=/gscratch/ark/rahuln/.cache',
       '--overwrite_cache',
       '--per_device_eval_batch_size=4']

# print command to log file
print(' '.join(cmd))
sys.stdout.flush()
sys.stderr.flush()
subprocess.call(cmd)

# write command out to file
with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
    cmd_str = ' '.join(cmd).replace(' --', ' \\\n    --')
    f.write(cmd_str)

