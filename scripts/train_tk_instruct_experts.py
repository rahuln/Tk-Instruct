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
parser.add_argument('--model_name_or_path', type=str,
                    default='allenai/tk-instruct-base-def-pos',
                    help='name of model to load from Huggingface, path to '
                         'model to load, or path to model soup info to '
                         'specify models to merge')
parser.add_argument('--train_on_dev', action='store_true',
                    help='train on dev set instead of test set')
parser.add_argument('--num_train_epochs', type=int, default=None,
                    help='number of epochs of training')
parser.add_argument('--max_steps', type=int, default=None,
                    help='maximum number of steps of training')
parser.add_argument('--per_device_batch_size', type=int, default=2,
                    help='per-GPU batch size for training')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                    help='number of gradient accumulation steps for training')
parser.add_argument('--max_num_instances_per_task', type=int, default=None,
                    help='maximum number of training instances per task')
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
if args.train_on_dev and data_dir.endswith('train'):
    data_dir = data_dir.replace('train', 'test')
use_dev = cfg.get('use_dev', False)
num_dev = cfg.get('num_dev', 50)

# get path to model or list of models to merge
merging_models = False
if os.path.exists(args.model_name_or_path):
    model_path = os.path.join(args.model_name_or_path, 'pytorch_model.bin')
    info_path = os.path.join(args.model_name_or_path, 'soup_info.json')
    if os.path.exists(model_path):
        # loading state_dict from file
        model_name_or_path = model_path
    elif os.path.exists(info_path):
        # merging models given a list of paths to their state_dicts
        base_path = 'results/niv2/tk-instruct-base/saved_model'
        with open(info_path, 'r') as f:
            soup_info = json.load(f)
        models_to_merge = list()
        for model_name in soup_info['models']:
            if model_name == 'allenai/tk-instruct-base-def-pos':
                models_to_merge.append(base_path)
            else:
                models_to_merge.append(model_name)
        models_to_merge = ','.join(models_to_merge)
        merging_models = True
    else:
        raise RuntimeError('`model_name_or_path` must be path to saved model '
                           'or soup_info.json file if merging models')
else:
    # load Huggingface model
    model_name_or_path = args.model_name_or_path

# create output directory
train_dir = 'train-dev' if args.train_on_dev else 'train'
train_dir += '-init-soup' if merging_models else ''
output_dir = os.path.join('results', dataset, 'tk-instruct-base-experts',
                          train_dir, args.exp_name, category)
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
       f'--data_dir={data_dir}/{category}',
       '--task_dir=data/tasks',
       f'--output_dir={output_dir}',
       '--cache_dir=/gscratch/ark/rahuln/.cache',
       '--overwrite_cache',
       f'--per_device_train_batch_size={args.per_device_batch_size}',
       f'--per_device_eval_batch_size={args.per_device_batch_size}',
       f'--gradient_accumulation_steps={args.gradient_accumulation_steps}',
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

# specify model to load or list of models to merge
if merging_models:
    cmd.extend(['--model_name_or_path=allenai/tk-instruct-base-def-pos',
                f'--models_to_merge={models_to_merge}'])
else:
    cmd.append(f'--model_name_or_path={model_name_or_path}')

# specify max_num_instances_per_task
if args.max_num_instances_per_task is not None:
    max_num_instances = args.max_num_instances_per_task
    cmd.append(f'--max_num_instances_per_task={max_num_instances}')

# set number of training epochs or steps
if args.num_train_epochs is not None:
    cmd.append(f'--num_train_epochs={args.num_train_epochs}')
elif args.max_steps is not None:
    cmd.append(f'--max_steps={args.max_steps}')

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

# remove checkpoints to save space
dirnames = glob(os.path.join(output_dir, 'checkpoint*'))
for dirname in dirnames:
    shutil.rmtree(dirname)

