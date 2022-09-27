""" script to train specialized experts initialized from tk-instruct-base
    on different task categories """

from argparse import ArgumentParser
from glob import glob
import json
import os
import shutil
import subprocess
import sys


# command-line arguments
parser = ArgumentParser()
parser.add_argument('cfg_file', type=str, help='path to config file')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='name of experiment (e.g., number of training steps)')
parser.add_argument('--train_on_dev', action='store_true',
                    help='train on dev set instead of test set')
parser.add_argument('--num_train_epochs', type=int, default=None,
                    help='number of epochs of training')
parser.add_argument('--max_steps', type=int, default=None,
                    help='maximum number of steps of training')
parser.add_argument('--logging_steps', type=int, default=10,
                    help='number of steps between logging outputs')
parser.add_argument('--eval_steps', type=int, default=0,
                    help='number of steps before each evaluation')
parser.add_argument('--save_steps', type=int, default=500,
                    help='number of steps between saves')
parser.add_argument('--index', type=int, default=None,
                    help='index of Slurm array job')
args = parser.parse_args()

# load config file, select task category using index from command line
with open(args.cfg_file, 'r') as f:
    cfg = json.load(f)

if args.train_on_dev:
    category = cfg['test_categories'][args.index]
else:
    category = cfg['categories'][args.index]
dataset = cfg.get('dataset', 'niv2')
data_dir = cfg.get('data_dir', 'data/splits/category')
use_dev = cfg.get('use_dev', False)

# create output directory
output_dir = os.path.join('results', dataset, 'tk-instruct-base-experts',
                          'train', args.exp_name, category)
os.makedirs(output_dir, exist_ok=True)

# get run name and number of training epochs/steps
run_name = cfg['run_name_fmt'].format(category=category)
if args.num_train_epochs is None and args.max_steps is None:
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
       f'--data_dir={data_dir}/{category}',
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
       f'--logging_steps={args.logging_steps}',
       f'--evaluation_strategy={"steps" if args.eval_steps > 0 else "no"}',
       f'--eval_steps={args.eval_steps}',
       '--save_strategy=steps',
       f'--save_steps={args.save_steps}',
       '--save_total_limit=2',
       f'--run_name={run_name}',
       f'--train_on_dev={args.train_on_dev}']

# set number of training epochs or steps
if args.num_train_epochs is not None:
    cmd.append(f'--num_train_epochs={args.num_train_epochs}')
elif args.max_steps is not None:
    cmd.append(f'--max_steps={args.max_steps}')

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

# remove checkpoints to save space
dirnames = glob(os.path.join(output_dir, 'checkpoint*'))
for dirname in dirnames:
    shutil.rmtree(dirname)

