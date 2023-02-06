""" script to split NaturalInstructionsV2 dataset into subdirectories by task
    category and filter to tasks with a minimum number of examples """

from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from itertools import chain
import json
import os


# command-line arguments
parser = ArgumentParser(description='script to split NaturalInstructionsV2 '
                                    'dataset into subdirectories by task '
                                    'category and filter to tasks with a '
                                    'minimum number of examples')
parser.add_argument('--outdir', type=str, default='data/splits/category',
                    help='output directory for task files')
parser.add_argument('--min-num-instances', type=int, default=200,
                    help='minimum number of instances per task')
parser.add_argument('--min-num-test-instances', type=int, default=100,
                    help='minimum number of instances per task in test set')
parser.add_argument('--min-num-tasks', type=int, default=10,
                    help='minimum number of tasks per category to be included '
                         'in the training set')
parser.add_argument('--num-eval-per-task', type=int, default=100,
                    help='number of dev/test instances per task')
parser.add_argument('--use-default-split', action='store_true',
                    help='restrict train/test task categories to those used '
                         'in default NaturalInstructionsV2 split')
parser.add_argument('--use-all-tasks-for-eval', action='store_true',
                    help='use all training tasks for dev and test sets in '
                         'training task directories, instead of just using '
                         'the tasks for that task category')
parser.add_argument('--include-tasks', type=str, default=None,
                    help='path to file with list of tasks to include (all '
                         'other tasks are excluded)')


if __name__ == '__main__':
    args = parser.parse_args()

    # get all task files
    file_idx = lambda x: int(x.split('/')[2].split('_')[0].replace('task', ''))
    files = sorted(glob('data/tasks/*.json'), key=file_idx)

    # limit to tasks that are in original train/test sets
    with open('data/splits/default/train_tasks.txt', 'r') as f:
        train_tasks = set([line.strip() for line in f.readlines()])
    with open('data/splits/default/test_tasks.txt', 'r') as f:
        test_tasks = set([line.strip() for line in f.readlines()])
    keep_tasks = train_tasks | test_tasks

    # construct map of task category to set of tasks
    category_to_tasks = defaultdict(lambda: list())
    num_inst_per_task = dict()
    for fname in files:
        with open(fname, 'r') as f:
            task = json.load(f)

        # filter out tasks that are not in original train/test sets
        task_name = os.path.basename(fname).replace('.json', '')
        if task_name not in keep_tasks:
            continue

        # filter out tasks with too few instances
        num_inst = len(task['Instances'])
        num_inst_per_task[task_name] = num_inst
        if args.use_default_split:
            if task_name in train_tasks and num_inst < args.min_num_instances:
                continue
            elif task_name in test_tasks and \
                 num_inst < args.min_num_test_instances:
                continue
        else:
            if num_inst < args.min_num_instances:
                continue

        # add task name to set for its category
        task_category = task['Categories'][0]
        category_to_tasks[task_category].append(task_name)

    # split into train/test sets based on number of datasets per category
    train_categories = {key : value for key, value in category_to_tasks.items()
                        if len(value) >= args.min_num_tasks}

    # do this for now, will filter for test task categories below
    test_categories = category_to_tasks

    # filter train/test categories to those in original train/test splits
    if args.use_default_split:
        train_categories = {key : value for key, value
                            in train_categories.items()
                            if all([task in train_tasks for task in value])}
        test_categories = {key : value for key, value
                           in test_categories.items()
                           if all([task in test_tasks for task in value])}

    # filter out tasks not in list of tasks to include
    if args.include_tasks is not None:
        with open(args.include_tasks, 'r') as f:
            tasks_to_include = set([line.strip() for line in f.readlines()])
        train_categories = {key : [task for task in value
                                   if task in tasks_to_include]
                            for key, value in train_categories.items()}
        test_categories = {key : [task for task in value
                                  if task in tasks_to_include]
                           for key, value in test_categories.items()}

    all_train_tasks = sorted(chain(*train_categories.values()))
    all_test_tasks = sorted(chain(*test_categories.values()))

    # construct train sub-directory, where each training set contains the tasks
    # in that category and each dev and test set contains all training tasks
    for category, tasks in train_categories.items():
        category_dir = category.lower().replace(' ', '_')
        savedir = os.path.join(args.outdir, 'train', category_dir)
        os.makedirs(savedir, exist_ok=True)
        with open(os.path.join(savedir, 'train_tasks.txt'), 'w') as f:
            f.write('\n'.join(tasks) + '\n')
        for subset in ('dev', 'test'):
            with open(os.path.join(savedir, f'{subset}_tasks.txt'), 'w') as f:
                if args.use_all_tasks_for_eval:
                    f.write('\n'.join(all_train_tasks) + '\n')
                else:
                    f.write('\n'.join(tasks) + '\n')

    # construct train sub-directory, where training, dev, and test sets
    # contain tasks across all train categories
    savedir = os.path.join(args.outdir, 'train', 'all')
    os.makedirs(savedir, exist_ok=True)
    for subset in ('train', 'dev', 'test'):
        with open(os.path.join(savedir, f'{subset}_tasks.txt'), 'w') as f:
            f.write('\n'.join(all_train_tasks) + '\n')

    # calculate relative scaling factors for training task categories based on
    # number of tasks and number of instances, output to file
    task_to_cat = {t : k for k, v in train_categories.items() for t in v}
    train_num_tasks = {k : len(v) for k, v in train_categories.items()}
    num_eval = args.num_eval_per_task
    train_num_inst = {k : sum([num_inst_per_task[task_name] - num_eval
                               for task_name in v])
                      for k, v in train_categories.items()}
    max_tasks = max(list(train_num_tasks.values()))
    max_inst = max(list(train_num_inst.values()))
    num_tasks_scales = {name : train_num_tasks[task_to_cat[name]] / max_tasks
                        for name in sorted(task_to_cat.keys())}
    num_inst_scales = {name : train_num_inst[task_to_cat[name]] / max_inst
                       for name in sorted(task_to_cat.keys())}
    traindir = os.path.join(args.outdir, 'train')
    with open(os.path.join(traindir, 'num_tasks_scales.json'), 'w') as f:
        json.dump(num_tasks_scales, f, indent=4)
    with open(os.path.join(traindir, 'num_inst_scales.json'), 'w') as f:
        json.dump(num_inst_scales, f, indent=4)

    # construct test sub-directory, where each training, dev, and test set
    # contains the tasks in that category
    for category, tasks in test_categories.items():
        category_dir = category.lower().replace(' ', '_')
        savedir = os.path.join(args.outdir, 'test', category_dir)
        os.makedirs(savedir, exist_ok=True)
        for subset in ('train', 'dev', 'test'):
            with open(os.path.join(savedir, f'{subset}_tasks.txt'), 'w') as f:
                f.write('\n'.join(tasks) + '\n')

    # construct test sub-directory, where training, dev, and test sets
    # contain tasks across all test categories
    savedir = os.path.join(args.outdir, 'test', 'all')
    os.makedirs(savedir, exist_ok=True)
    for subset in ('train', 'dev', 'test'):
        with open(os.path.join(savedir, f'{subset}_tasks.txt'), 'w') as f:
            f.write('\n'.join(all_test_tasks) + '\n')

    # print some statistics
    print('num training categories:', len(train_categories))
    print('largest training category (num tasks):',
          max(list(map(len, train_categories.values()))))
    print('num training tasks:', len(all_train_tasks))
    print('num test categories:', len(test_categories))
    print('largest test category (num tasks):',
          max(list(map(len, test_categories.values()))))
    print('num test tasks:', len(all_test_tasks))

