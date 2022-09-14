""" script to construct and evaluate greedy soup (with replacement) on a
    specific test task category """

from glob import glob
import json
import os
import subprocess
import sys


# load config file, select task category using index from command line
with open(sys.argv[-2], 'r') as f:
    cfg = json.load(f)
category = cfg['test_categories'][int(sys.argv[-1])]
data_dir = cfg.get('eval_data_dir', 'data/splits/default')
dataset = cfg.get('dataset', 'natural-instructions-v2')
max_soup_size = cfg.get('max_soup_size', 10)
include_base_model = cfg.get('include_base_model', False)
start_with_base_model = cfg.get('start_with_base_model', False)
use_dev = cfg.get('use_dev', False)

# specify path to models to use as soup components
path_to_soup_components = os.path.join('results', dataset,
                                       'tk-instruct-base-experts', 'train',
                                       cfg['exp_name'])

# create output directory
output_dir = os.path.join('results', dataset, 'tk-instruct-base-experts',
                          'evaluate-greedy-soup', cfg['exp_name'], category)
os.makedirs(output_dir, exist_ok=True)

# check for existing results
if os.path.exists(os.path.join(output_dir, 'metrics.json')):
    print('results already exist, exiting...')
    sys.exit()

# set up command
cmd = ['python', 'src/run_greedy_soup.py',
       '--do_train',
       '--do_predict',
       '--predict_with_generate',
       '--evaluation_strategy=no',
       '--model_name_or_path=allenai/tk-instruct-base-def-pos',
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
       f'--data_dir={data_dir}/{category}',
       '--task_dir=data/tasks',
       f'--output_dir={output_dir}',
       '--overwrite_output_dir',
       '--cache_dir=/gscratch/ark/rahuln/.cache',
       '--overwrite_cache',
       '--per_device_eval_batch_size=4',
       f'--path_to_soup_components={path_to_soup_components}',
       f'--max_soup_size={max_soup_size}',
       f'--include_base_model={include_base_model}',
       f'--start_with_base_model={start_with_base_model}']

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

