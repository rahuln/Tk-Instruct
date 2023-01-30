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
parser.add_argument('--base_model', type=str,
                    default='allenai/tk-instruct-base-def-pos',
                    help='name of or path to base model')
parser.add_argument('--include_base_model', action='store_true',
                    help='include base model as possible soup component')
parser.add_argument('--start_with_base_model', action='store_true',
                    help='use base model as initial soup component')
parser.add_argument('--finetuned_model_path', type=str, default=None,
                    help='include model fine-tuned on target task as '
                         'possible soup component')
parser.add_argument('--max_num_instances_per_task', type=int, default=None,
                    help='maximum number of training instances per task')
parser.add_argument('--output_ensemble', action='store_true',
                    help='use output ensemble instead of parameter averaging')
parser.add_argument('--use_train_as_dev', action='store_true',
                    help='use training set as validation set')
parser.add_argument('--use_test_as_dev', action='store_true',
                    help='use test set as validation set')
parser.add_argument('--use_dev', action='store_true',
                    help='use dev set, overriding value in config file')
parser.add_argument('--num_dev', type=int, default=None,
                    help='number of dev set examples, overrides config')
parser.add_argument('--eval_on_task', type=str, default=None,
                    help='indicates that data_dir contains set of tasks and '
                         'evaluation run should evaluate on tasks rather than '
                         'task categories')
parser.add_argument('--instance_ids_dir', type=str, default=None,
                    help='path to directory containing files for each task '
                         'with instance IDs to use as dev set for that task')
parser.add_argument('--param_groups', type=str, default=None,
                    help='regex patterns for parameter groups')
parser.add_argument('--num_experts', type=int, default=None,
                    help='number of randomly selected experts to use instead '
                         'of the full set')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--suffix', type=str, default=None,
                    help='suffix to add to name of results directory')
parser.add_argument('--index', type=int, default=None,
                    help='index of Slurm array job')
args = parser.parse_args()


# mapping between Huggingface model names and output directory names
model_to_dirname = {
    'allenai/tk-instruct-base-def-pos' : 'tk-instruct-base',
    'google/t5-base-lm-adapt' : 't5-base-lm-adapt',
}

# check to make sure we're evaluating at the task level if restricting to
# certain set of instance IDs for each task
if args.instance_ids_dir is not None and args.eval_on_task is None:
    raise ValueError('instance_ids_dir can only be specified if '
                     'eval_on_task is not None')

# load config file, select task category using index from command line
with open(args.cfg_file, 'r') as f:
    cfg = json.load(f)

if args.eval_on_task is not None:
    with open(args.eval_on_task, 'r') as f:
        tasks = sorted([line.strip() for line in f.readlines()])
    category = tasks[args.index]
else:
    category = cfg['test_categories'][args.index]
dataset = cfg.get('dataset', 'niv2')
use_dev = args.use_dev or cfg.get('use_dev', False)
num_dev = args.num_dev if args.num_dev is not None else cfg.get('num_dev', 50)

# get model directory name from base model
if args.base_model in model_to_dirname:
    model_dirname_prefix = model_to_dirname[args.base_model]
else:
    model_dirname_prefix = model_name_or_path.split('/')[-1]
model_dirname = f'{model_dirname_prefix}-experts'

# specify path to models to use as soup components
path_to_soup_components = os.path.join('results', dataset, model_dirname,
                                       'train', args.exp_name)

# construct directory for greedy soup results
resdir = 'output-ensemble' if args.output_ensemble else 'greedy-soup'
if args.start_with_base_model:
    resdir += '-init-base'
elif args.include_base_model:
    resdir += '-include-base'
if args.use_train_as_dev:
    resdir += '-train-as-dev'
if args.use_test_as_dev:
    resdir += '-test-as-dev'
if args.eval_on_task is not None:
    resdir += '-eval-task'
if args.finetuned_model_path is not None:
    resdir += '-incl-ft-model'
if args.instance_ids_dir is not None:
    inst_ids_dirname = os.path.basename(args.instance_ids_dir)
    resdir += f'-inst-ids/{inst_ids_dirname}'
if args.suffix is not None:
    resdir += f'-{args.suffix}'

# create output directory
num_dev_suffix = f'-dev-{num_dev}' if args.num_dev is not None else ''
output_dir = os.path.join('results', dataset, model_dirname, 'evaluate',
                          resdir, args.exp_name + num_dev_suffix, category)
os.makedirs(output_dir, exist_ok=True)

# construct data directory depending on experiment settings
if args.instance_ids_dir is not None:
    data_dir = args.data_dir
else:
    data_dir = os.path.join(args.data_dir, category)

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
       f'--model_name_or_path={args.base_model}',
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
       '--per_device_eval_batch_size=4',
       f'--seed={args.seed}',
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
    cmd.extend(['--do_eval', '--use_dev', f'--num_dev={num_dev}',
                f'--use_train_as_dev={args.use_train_as_dev}',
                f'--use_test_as_dev={args.use_test_as_dev}'])

# use output ensemble instead of parameter averaging
if args.output_ensemble:
    cmd.append('--output_ensemble')

# include fine-tuned model as possible soup component
if args.finetuned_model_path is not None:
    finetuned_model = os.path.join(args.finetuned_model_path, category,
                                   'pytorch_model.bin')
    cmd.append(f'--include_models={finetuned_model}')

# specify file containing dev instance IDs as well as test task name for
# filtering dev/test instances
if args.instance_ids_dir is not None:
    cmd.extend([f'--eval_instance_ids_file={args.instance_ids_dir}/{category}.txt',
                f'--test_task={category}'])

# specify parameter groups
if args.param_groups is not None:
    cmd.append(f'--param_groups={args.param_groups}')

# specify number of randomly selected experts
if args.num_experts is not None:
    cmd.append(f'--num_experts={args.num_experts}')

# print command to log file
print(' '.join(cmd))
sys.stdout.flush()
sys.stderr.flush()
subprocess.call(cmd)

# write command out to file
with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
    cmd_str = ' '.join(cmd).replace(' --', ' \\\n    --')
    f.write(cmd_str)

