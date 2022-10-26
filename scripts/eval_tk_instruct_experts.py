""" script to evaluate specialized experts initialized from tk-instruct-base
    on specified task categories """

from argparse import ArgumentParser
from glob import glob
import json
import os
import sys
import subprocess

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
parser.add_argument('--max_num_instances_per_task', type=int, default=None,
                    help='maximum number of training instances per task')
parser.add_argument('--random_merge', type=int, default=0,
                    help='evaluate by merging the specified number of '
                         'experts, randomly selected from trained experts')
parser.add_argument('--seed', type=int, default=42,
                    help='seed for selecting random set of experts')
parser.add_argument('--index', type=int, default=None,
                    help='index of Slurm array job')
args = parser.parse_args()

# load config file, select task category using index from command line
with open(args.cfg_file, 'r') as f:
    cfg = json.load(f)
if args.random_merge > 0:
    categories = cfg['test_categories']
else:
    categories = cfg['categories']
category = categories[args.index]
dataset = cfg.get('dataset', 'niv2')
use_dev = cfg.get('use_dev', False)
num_dev = cfg.get('num_dev', 50)

# specify model path
if args.random_merge > 0:
    np.random.seed(args.seed)
    path_to_soup_components = os.path.join('results', dataset,
                                           'tk-instruct-base-experts',
                                           'train', args.exp_name)
    experts = sorted(glob(os.path.join(path_to_soup_components, '*')))
    idx = np.random.choice(len(experts), size=args.random_merge, replace=False)
    models_to_merge = ','.join([experts[i] for i in idx])
    data_dir = os.path.join(args.data_dir, category)
else:
    model_name_or_path = os.path.join('results', dataset,
                                      'tk-instruct-base-experts', 'train',
                                      args.exp_name, category)
    data_dir = args.data_dir

# create output directory
output_dir = os.path.join('results', dataset, 'tk-instruct-base-experts',
                          'evaluate', args.eval_dirname, args.exp_name,
                          category)
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
       f'--data_dir={data_dir}',
       '--task_dir=data/tasks',
       f'--output_dir={output_dir}',
       '--overwrite_output_dir',
       '--cache_dir=/gscratch/ark/rahuln/.cache',
       '--overwrite_cache',
       '--per_device_eval_batch_size=4']

# specify model(s) to use
if args.random_merge > 0:
    cmd.extend(['--model_name_or_path=allenai/tk-instruct-base-def-pos',
                f'--models_to_merge={models_to_merge}'])
else:
    cmd.append(f'--model_name_or_path={model_name_or_path}')

# specify max_num_instances_per_task
if args.max_num_instances_per_task is not None:
    max_num_instances = args.max_num_instances_per_task
    cmd.append(f'--max_num_instances_per_task={max_num_instances}')

# use dev/test split of test set if specified
if use_dev:
    cmd.extend(['--do_eval', '--use_dev', f'--num_dev={num_dev}'])

# print command to log file
print(' '.join(cmd))
sys.stdout.flush()
sys.stderr.flush()
subprocess.call(cmd)

# write command out to file
with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
    cmd_str = ' '.join(cmd).replace(' --', ' \\\n    --')
    f.write(cmd_str)

