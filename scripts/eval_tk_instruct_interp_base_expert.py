""" script to evaluate interpolation between tk-instruct-base and specialized
    expert on a given dataset """

from glob import glob
from itertools import product
import json
import os
import subprocess
import sys

import numpy as np


# load config file, select task category using index from command line
with open(sys.argv[-2], 'r') as f:
    cfg = json.load(f)
weights = np.linspace(0, 1, cfg.get('num_weights', 11))
grid = list(product(cfg['categories'], weights))
category, weight = grid[int(sys.argv[-1])]
data_dir = cfg.get('eval_data_dir', 'data/splits/default')
dataset = cfg.get('dataset', 'natural-instructions-v2')

# specify model path
model_name_or_path = os.path.join('results', dataset,
                                  'tk-instruct-base-experts', 'train',
                                  cfg['exp_name'], category)

# specify path to base model
base_model_path = os.path.join('results', dataset, 'tk-instruct-base',
                               'saved_model')

# construct strings for paths to models to merge and merging weights
models_to_merge = model_name_or_path + ',' + base_model_path
merging_weights = str(weight) + ',' + str(1 - weight)

# create output directory
output_dir = os.path.join('results', dataset, 'tk-instruct-base-experts',
                          cfg['eval_type'], cfg['exp_name'], category,
                          f'weight-{weight:.1f}')
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
       '--max_num_instances_per_task=100',
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

# print command to log file
print(' '.join(cmd))
sys.stdout.flush()
sys.stderr.flush()
subprocess.call(cmd)

# write command out to file
with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
    cmd_str = ' '.join(cmd).replace(' --', ' \\\n    --')
    f.write(cmd_str)

