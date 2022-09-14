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
parser.add_argument('--use-default-split', action='store_true',
                    help='restrict train/test task categories to those used '
                         'in default NaturalInstructionsV2 split')


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
    for fname in files:
        with open(fname, 'r') as f:
            task = json.load(f)

        # filter out tasks that are not in original train/test sets
        task_name = os.path.basename(fname).replace('.json', '')
        if task_name not in keep_tasks:
            continue

        # filter out tasks with too few instances
        num_inst = len(task['Instances'])
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
    test_categories = {key : value for key, value in category_to_tasks.items()
                       if len(value) < args.min_num_tasks}

    # filter train/test categories to those in original train/test splits
    if args.use_default_split:
        train_categories = {key : value for key, value
                            in train_categories.items()
                            if all([task in train_tasks for task in value])}
        test_categories = {key : value for key, value
                           in test_categories.items()
                           if all([task in test_tasks for task in value])}

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
                f.write('\n'.join(all_train_tasks) + '\n')

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

