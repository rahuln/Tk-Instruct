""" for a given set of embeddings, compute the set of most similar training
    task instances using a set of test task instances and save the instance
    IDs to a file """

from argparse import ArgumentParser
from glob import glob
from itertools import chain
import json
import os

import torch
from tqdm import tqdm


# command-line arguments
parser = ArgumentParser(description='compute and save similar instances '
                                    'based on instance embeddings')
parser.add_argument('emb_file', type=str, help='path to instance embeddings')
parser.add_argument('--cfg_file', type=str,
                    default='scripts/config/'
                            'category_ts10_tr200_ev100_dev50.json',
                    help='path to config file for data split')
parser.add_argument('--task_dir', type=str, default='data/tasks',
                    help='directory containign task info JSON files')
parser.add_argument('--num_per_instance', type=int, default=20,
                    help='number of most similar training instances per '
                         'test instance to use')
parser.add_argument('--savedir', type=str, default='tmp',
                    help='output directory for IDs of similar instances')


def main(args):
    """ main script """

    # load embeddings file and config file, get train/test category names
    emb = torch.load(args.emb_file)
    with open(args.cfg_file, 'r') as f:
        cfg = json.load(f)
    train_categories = set(cfg['categories'])
    test_categories = set(cfg['test_categories'])

    # construct mapping from task name to category
    task_to_category = {task : category for task, category in 
                        zip(emb['task_names'], emb['task_categories'])}

    # get names of train and test tasks, instance IDs of training tasks, and
    # training tasks' instance embeddings
    task_names = sorted(emb['task_names'])
    train_tasks = [task for task in task_names
                   if task_to_category[task] in train_categories]
    train_instance_ids = list(chain(*[emb['task_instance_ids'][task]
                                      for task in train_tasks]))
    train_emb = torch.cat([emb['embeddings'][task] for task in train_tasks])
    test_tasks = [task for task in task_names
                  if task_to_category[task] in test_categories]

    # iterate through test tasks, find most similar instances among training
    # tasks and keep track of them
    similar_instances = dict()
    for task in tqdm(test_tasks, desc='finding similar instances'):
        test_emb = emb['embeddings'][task]
        sim = torch.matmul(test_emb, train_emb.t())
        most_sim = torch.argsort(sim, dim=1, descending=True)
        idxs = most_sim[:, :args.num_per_instance].flatten()
        sim_ids = sorted(set([train_instance_ids[idx] for idx in idxs]))
        similar_instances[task] = sim_ids

    # save results
    if args.savedir is not None:
        embname, _ = os.path.splitext(os.path.basename(args.emb_file))
        subdir = f'{embname}-top-{args.num_per_instance}-per-instance'
        os.makedirs(os.path.join(args.savedir, subdir), exist_ok=True)
        for task, ids in similar_instances.items():
            savename = os.path.join(args.savedir, subdir, f'{task}.txt')
            with open(savename, 'w') as f:
                f.write('\n'.join(ids))


if __name__ == '__main__':
    main(parser.parse_args())
