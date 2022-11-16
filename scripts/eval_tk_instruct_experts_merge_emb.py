""" script to merge trained experts based on similarity of task embeddings
    to a target task category and evaluate on that target category """

from argparse import ArgumentParser
from glob import glob
import json
import os
import sys
import subprocess

import numpy as np
import torch


# command-line arguments
parser = ArgumentParser()
parser.add_argument('cfg_file', type=str, help='path to config file')
parser.add_argument('emb_file', type=str,
                    help='path to file with task category embeddings')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='name of experiment (e.g., number of training steps)')
parser.add_argument('--base_model', type=str,
                    default='allenai/tk-instruct-base-def-pos',
                    help='name of or path to base model')
parser.add_argument('--include_base_model_in_merge', action='store_true',
                    help='include base model when merging experts')
parser.add_argument('--data_dir', type=str, default='data/splits/default',
                    help='data directory for evaluation tasks')
parser.add_argument('--max_num_instances_per_task', type=int, default=None,
                    help='maximum number of training instances per task')
parser.add_argument('--task_embeddings', action='store_true',
                    help='embeddings are for each task, not category')
parser.add_argument('--num_experts_to_merge', type=int, default=3,
                    help='number of most-similar experts to merge')
parser.add_argument('--suffix', type=str, default=None,
                    help='suffix to add to results directory name '
                         '(e.g., what type of embeddings were used)')
parser.add_argument('--index', type=int, default=None,
                    help='index of Slurm array job')
args = parser.parse_args()

# mapping between Huggingface model names and output directory names
model_to_dirname = {
    'allenai/tk-instruct-base-def-pos' : 'tk-instruct-base',
    'google/t5-base-lm-adapt' : 't5-base-lm-adapt',
}

# load config file, select task category using index from command line
with open(args.cfg_file, 'r') as f:
    cfg = json.load(f)
test_categories = cfg['test_categories']
if args.task_embeddings:
    tasks = sorted(os.listdir(args.data_dir))
    category = tasks[args.index]
else:
    category = test_categories[args.index]
dataset = cfg.get('dataset', 'niv2')
use_dev = cfg.get('use_dev', False)
num_dev = cfg.get('num_dev', 50)

# load embeddings file, find most similar experts based on embeddings
emb = torch.load(args.emb_file)
if args.task_embeddings:
    # embeddings are specific to each task, not category
    train_idx = [idx for idx, cat in enumerate(emb['task_categories'])
                 if cat not in test_categories]
    train_emb = emb['embeddings'][train_idx, :]
    train_categories = [emb['task_categories'][idx] for idx in train_idx]
    target_emb = emb['embeddings'][emb['task_names'].index(category)]
    cosine_sim = torch.matmul(train_emb, target_emb)
    most_sim = torch.argsort(cosine_sim, descending=True)
    merge_categories = [train_categories[idx] for idx in
                        most_sim[:args.num_experts_to_merge]]
    merge_categories = sorted(set(merge_categories))
else:
    # embeddings are mean embeddings over task categories
    train_idx = [idx for idx, cat in enumerate(emb['categories'])
                 if cat not in test_categories]
    train_emb = emb['embeddings'][train_idx, :]
    train_categories = [emb['categories'][idx] for idx in train_idx]
    target_emb = emb['embeddings'][emb['categories'].index(category)]
    cosine_sim = torch.matmul(train_emb, target_emb)
    most_sim = torch.argsort(cosine_sim, descending=True)
    merge_categories = [train_categories[idx] for idx in
                        most_sim[:args.num_experts_to_merge]]

# get model directory name from base model
if args.base_model in model_to_dirname:
    model_dirname_prefix = model_to_dirname[args.base_model]
else:
    model_dirname_prefix = model_name_or_path.split('/')[-1]
model_dirname = f'{model_dirname_prefix}-experts'

# specify paths to models to be merged
models_to_merge = list()
for cat in merge_categories:
    model_name_or_path = os.path.join('results', dataset, model_dirname,
                                      'train', args.exp_name, cat)
    models_to_merge.append(model_name_or_path)
models_to_merge = ','.join(models_to_merge)

# create output directory
suffix = f'-{args.suffix}' if args.suffix is not None else ''
resdir = f'merge-top-{args.num_experts_to_merge}{suffix}'
if args.task_embeddings:
    resdir += '-eval-on-task'
output_dir = os.path.join('results', dataset, model_dirname, 'evaluate',
                          resdir, args.exp_name, category)
os.makedirs(output_dir, exist_ok=True)

# check for existing results
if os.path.exists(os.path.join(output_dir, 'metrics.json')):
    print('results already exist, exiting...')
    sys.exit()

# set up command
cmd = ['python', 'src/run_s2s.py',
       f'--model_name_or_path={args.base_model}',
       f'--models_to_merge={models_to_merge}',
       f'--include_base_model_in_merge={args.include_base_model_in_merge}',
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
       f'--data_dir={args.data_dir}/{category}',
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

