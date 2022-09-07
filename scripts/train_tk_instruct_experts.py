""" script to train specialized experts initialized from tk-instruct-base
    on different task categories """

from glob import glob
import json
import os
import shutil
import subprocess
import sys


# load config file, select task category using index from command line
with open(sys.argv[-2], 'r') as f:
    cfg = json.load(f)
category = cfg['categories'][int(sys.argv[-1])]
dataset = cfg.get('dataset', 'natural-instructions-v2')

# create output directory
output_dir = os.path.join('results', dataset, 'tk-instruct-base-experts',
                          cfg['exp_name'], category)
os.makedirs(output_dir, exist_ok=True)

# get run name and number of training epochs/steps
run_name = cfg['run_name_fmt'].format(category=category)
num_train_epochs = cfg.get('num_train_epochs', None)
max_steps = cfg.get('max_steps', None)
if num_train_epochs is None and max_steps is None:
    raise RuntimeError('must specify either num_train_epochs or max_steps')

# check for existing results
if os.path.exists(os.path.join(output_dir, 'metrics.json')):
    print('results already exist, exiting...')
    sys.exit()

# set up command
cmd = ['python', 'src/run_s2s.py',
       '--do_train',
       '--do_predict',
       '--predict_with_generate',
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
       f'--data_dir=data/splits/category/{category}',
       '--task_dir=data/tasks',
       f'--output_dir={output_dir}',
       '--overwrite_output_dir',
       '--cache_dir=/gscratch/ark/rahuln/.cache',
       '--overwrite_cache',
       '--per_device_train_batch_size=2',
       '--per_device_eval_batch_size=2',
       '--gradient_accumulation_steps=8',
       '--learning_rate=5e-05',
       '--lr_scheduler_type=constant',
       '--warmup_steps=0',
       '--logging_strategy=steps',
       f'--logging_steps={cfg["logging_steps"]}',
       '--evaluation_strategy=no',
       '--save_strategy=steps',
       f'--save_steps={cfg["save_steps"]}',
       '--save_total_limit=2',
       f'--run_name={run_name}']

# set number of training epochs or steps
if num_train_epochs is not None:
    cmd.append(f'--num_train_epochs={num_train_epochs}')
elif max_steps is not None:
    cmd.append(f'--max_steps={max_steps}')

# print command to log file
print(' '.join(cmd))
sys.stdout.flush()
sys.stderr.flush()
subprocess.call(cmd)

# write command out to file
with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
    cmd_str = ' '.join(cmd).replace(' --', ' \\\n    --')
    f.write(cmd_str)

# remove checkpoints to save space
dirnames = glob(os.path.join(output_dir, 'checkpoint*'))
for dirname in dirnames:
    shutil.rmtree(dirname)

