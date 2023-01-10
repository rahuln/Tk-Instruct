""" script to merge trained experts based on relative probabilities of a task's
    prompt under various trained autoregressive LM experts """

from argparse import ArgumentParser
from glob import glob
import json
import os
import sys
import subprocess

import numpy as np
from scipy.special import softmax
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# command-line arguments
parser = ArgumentParser()
parser.add_argument('cfg_file', type=str, help='path to config file')
parser.add_argument('path_to_likelihood_models', type=str,
                    help='path to directory with autoregressive LM experts')
parser.add_argument('path_to_experts', type=str,
                    help='path to directory with experts to be merged')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='name of experiment (e.g., number of training steps)')
parser.add_argument('--likelihood_model', type=str, default='gpt2-large',
                    help='name of or path to autoregressive LM model')
parser.add_argument('--base_model', type=str,
                    default='allenai/tk-instruct-base-def-pos',
                    help='name of or path to base model')
parser.add_argument('--data_dir', type=str, default='data/splits/default',
                    help='data directory for evaluation tasks')
parser.add_argument('--task_file', type=str,
                    default='data/splits/category-ts10-tr200-ev100/test/all/'
                            'train_tasks.txt',
                    help='file containing names of tasks to evaluate on')
parser.add_argument('--task_dir', type=str, default='data/tasks',
                    help='path to directory with task info files')
parser.add_argument('--max_num_instances_per_task', type=int, default=None,
                    help='maximum number of training instances per task')
parser.add_argument('--num_experts_to_merge', type=int, default=-1,
                    help='number of highest-likelihood experts to merge')
parser.add_argument('--use_merging_weights', action='store_true',
                    help='use weighted merge (instead of uniform average)')
parser.add_argument('--suffix', type=str, default=None,
                    help='suffix to add to results directory name '
                         '(e.g., what type of embeddings were used)')
parser.add_argument('--cache_dir', type=str,
                    default='/gscratch/ark/rahuln/.cache',
                    help='path to Huggingface cache directory')
parser.add_argument('--index', type=int, default=None,
                    help='index of Slurm array job')
args = parser.parse_args()

# mapping between Huggingface model names and output directory names
model_to_dirname = {
    'allenai/tk-instruct-base-def-pos' : 'tk-instruct-base',
    'google/t5-base-lm-adapt' : 't5-base-lm-adapt',
}

# load config file, get settings from config
with open(args.cfg_file, 'r') as f:
    cfg = json.load(f)
dataset = cfg.get('dataset', 'niv2')
use_dev = cfg.get('use_dev', False)
num_dev = cfg.get('num_dev', 50)

# select task using index from command line
if args.task_file is not None:
    with open(args.task_file, 'r') as f:
        tasks = [line.strip() for line in f.readlines()]
else:
    tasks = sorted(os.listdir(args.data_dir))
task = tasks[args.index]

# find autoregressive LM experts used to calculate likelihood of prompt
files = sorted(glob(os.path.join(args.path_to_likelihood_models, '**',
                                 'pytorch_model.bin'), recursive=True))
likelihood_model_paths = [os.path.dirname(fname) for fname in files]

# find experts that will be merged, check that they match likelihood models
files = sorted(glob(os.path.join(args.path_to_experts, '**',
                                 'pytorch_model.bin'), recursive=True))
expert_model_paths = [os.path.dirname(fname) for fname in files]
assert len(likelihood_model_paths) == len(expert_model_paths), 'lengths'
for lik_path, exp_path in zip(likelihood_model_paths, expert_model_paths):
    assert os.path.basename(lik_path) == os.path.basename(exp_path), 'names'

# load info for task, get task definition
with open(os.path.join(args.task_dir, f'{task}.json'), 'r') as f:
    task_info = json.load(f)
defn = f'Definition: {task_info["Definition"][0]}'

# for each autoregressive LM expert, calculate the probability of the task
# definition
tokenizer = AutoTokenizer.from_pretrained(args.likelihood_model,
                                          cache_dir=args.cache_dir)
losses = np.zeros(len(likelihood_model_paths))
desc = 'calculating defn losses'
for i, dirname in enumerate(tqdm(likelihood_model_paths, desc=desc)):
    model = AutoModelForCausalLM.from_pretrained(dirname).cuda()
    inputs = tokenizer(defn, return_tensors='pt').to('cuda')
    inputs['labels'] = inputs['input_ids']
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
    losses[i] = outputs.loss.item()
merging_weights = np.exp(-losses)   # convert neg log-probs to probs

# specify paths to models to be merged and merging weights
if args.num_experts_to_merge == -1: # merge all experts
    models_to_merge = ','.join(expert_model_paths)
    merging_weights = softmax(merging_weights)
else:
    indices = np.argsort(merging_weights)[::-1]   # sort in descending order
    models_to_merge = [expert_model_paths[idx] for idx
                       in indices[:args.num_experts_to_merge]]
    models_to_merge = ','.join(models_to_merge)
    merging_weights = [merging_weights[idx] for idx
                       in indices[:args.num_experts_to_merge]]
    merging_weights = softmax(merging_weights)

# get model directory name from base model
if args.base_model in model_to_dirname:
    model_dirname_prefix = model_to_dirname[args.base_model]
else:
    model_dirname_prefix = model_name_or_path.split('/')[-1]
model_dirname = f'{model_dirname_prefix}-experts'

# create output directory
resdir = f'likelihood-merge/'
if args.num_experts_to_merge == -1:
    resdir += 'all-experts'
else:
    resdir += f'top-{args.num_experts_to_merge}-experts'
if args.use_merging_weights:
    resdir += '-weighted-merge'
suffix = f'-{args.suffix}' if args.suffix is not None else ''
resdir = resdir + suffix
output_dir = os.path.join('results', dataset, model_dirname, 'evaluate',
                          resdir, args.exp_name, task)
os.makedirs(output_dir, exist_ok=True)

# check for existing results
if os.path.exists(os.path.join(output_dir, 'metrics.json')):
    print('results already exist, exiting...')
    sys.exit()

# set up command
cmd = ['python', 'src/run_s2s.py',
       f'--model_name_or_path={args.base_model}',
       f'--models_to_merge={models_to_merge}',
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
       f'--data_dir={args.data_dir}/{task}',
       f'--task_dir={args.task_dir}',
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

# specify merging weights
if args.use_merging_weights:
    merging_weights_str = ','.join(list(map(str, merging_weights)))
    cmd.append(f'--merging_weights={merging_weights_str}')

# print command to log file
print(' '.join(cmd))
sys.stdout.flush()
sys.stderr.flush()
subprocess.call(cmd)

# write command out to file
with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
    cmd_str = ' '.join(cmd).replace(' --', ' \\\n    --')
    f.write(cmd_str)

