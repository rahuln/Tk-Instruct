""" script to construct and evaluate greedy soup (with replacement) on a
    specific test task category """

from argparse import ArgumentParser
from glob import glob
import json
import os
import subprocess
import sys


# command-line arguments
parser = ArgumentParser()
parser.add_argument('cfg_file', type=str, help='path to config file')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='name of experiment (e.g., number of training steps)')
parser.add_argument('--data_dir', type=str, default='data/splits/default',
                    help='data directory for evaluation tasks')
parser.add_argument('--max_soup_size', type=int, default=10,
                    help='maximum number of allowed soup components')
parser.add_argument('--include_base_model', action='store_true',
                    help='include base model as possible soup component')
parser.add_argument('--start_with_base_model', action='store_true',
                    help='use base model as initial soup component')
parser.add_argument('--max_num_instances_per_task', type=int, default=None,
                    help='maximum number of training instances per task')
parser.add_argument('--output_ensemble', action='store_true',
                    help='use output ensemble instead of parameter averaging')
parser.add_argument('--index', type=int, default=None,
                    help='index of Slurm array job')
args = parser.parse_args()

# load config file, select task category using index from command line
with open(args.cfg_file, 'r') as f:
    cfg = json.load(f)
category = cfg['test_categories'][args.index]
dataset = cfg.get('dataset', 'niv2')
use_dev = cfg.get('use_dev', False)
num_dev = cfg.get('num_dev', 50)

# specify path to models to use as soup components
path_to_soup_components = os.path.join('results', dataset,
                                       'tk-instruct-base-experts', 'train',
                                       args.exp_name)

# construct directory for greedy soup results
resdir = 'output-ensemble' if args.output_ensemble else 'greedy-soup'
if args.start_with_base_model:
    resdir += '-init-base'
elif args.include_base_model:
    resdir += '-include-base'

# create output directory
output_dir = os.path.join('results', dataset, 'tk-instruct-base-experts',
                          'evaluate', resdir, args.exp_name, category)
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
       '--per_device_eval_batch_size=4',
       f'--path_to_soup_components={path_to_soup_components}',
       f'--max_soup_size={args.max_soup_size}',
       f'--include_base_model={args.include_base_model}',
       f'--start_with_base_model={args.start_with_base_model}']

# specify max_num_instances_per_task
if args.max_num_instances_per_task is not None:
    max_num_instances = args.max_num_instances_per_task
    cmd.append(f'--max_num_instances_per_task={max_num_instances}')

# use dev/test split of test set if specified
if use_dev:
    cmd.extend(['--do_eval', '--use_dev', f'--num_dev={num_dev}'])

# use output ensemble instead of parameter averaging
if args.output_ensemble:
    cmd.append('--output_ensemble')

# print command to log file
print(' '.join(cmd))
sys.stdout.flush()
sys.stderr.flush()
subprocess.call(cmd)

# write command out to file
with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
    cmd_str = ' '.join(cmd).replace(' --', ' \\\n    --')
    f.write(cmd_str)

