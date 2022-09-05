""" script to evaluate specialized experts initialized from tk-instruct-base
    on specified task categories """

from glob import glob
import json
import os
import sys
import subprocess


# load config file, select task category using index from command line
with open(sys.argv[-2], 'r') as f:
    cfg = json.load(f)
category = cfg['categories'][int(sys.argv[-1])]
data_dir = cfg.get('data_dir', 'data/splits/default')

# specify model path
model_name_or_path = os.path.join('results', 'natural-instructions-v2',
                                  'tk-instruct-base-experts', 'train',
                                  cfg['exp_name'], category)

# create output directory
output_dir = os.path.join('results', 'natural-instructions-v2',
                          'tk-instruct-base-experts', cfg['eval_type'],
                          cfg['exp_name'], category)
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
       f'--model_name_or_path={model_name_or_path}',
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

