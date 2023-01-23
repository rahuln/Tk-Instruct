""" train pretrained language model as classifier for predicting task category
    from task instruction for a set of tasks in the SuperNaturalInstructions
    dataset """

from argparse import ArgumentParser
from dataclasses import dataclass, field
import json
import os
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm


@dataclass
class CustomArguments:
    """ custom command-line arguments """

    tasks_dir: Optional[str] = field(
        default='data/tasks',
        metadata={'help': 'path to JSON info files for tasks'},
    )
    task_file: Optional[str] = field(
        default='data/splits/category-ts10-tr200-ev100/train/all/train_tasks.txt',
        metadata={'help': 'path to file with list of tasks'},
    )
    model_name: Optional[str] = field(
        default='roberta-base',
        metadata={'help': 'name of Huggingface model to fine-tune'},
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={'help': 'maximum length of input sequence'},
    )
    cache_dir: Optional[str] = field(
        default='/gscratch/ark/rahuln/.cache',
        metadata={'help': 'path to Huggingface cache directory'},
    )
    split_seed: Optional[int] = field(
        default=42,
        metadata={'help' : 'random seed for train/dev/test split'},
    )


class TaskCategoryDataset(torch.utils.data.Dataset):
    """ dataset used for multi-class classification of task category
        from task instruction for SuperNaturalInstructions tasks """

    def __init__(self, tasks_dir, task_file, tokenizer, max_length=512,
                 subset='train', seed=42):
        super(TaskCategoryDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length

        # load list of tasks
        with open(task_file, 'r') as f:
            tasks = [line.strip() for line in f.readlines()]

        # get task instruction and category for each task
        self.instructions = list()
        self.categories = list()
        for task in tqdm(tasks, desc='loading task info'):
            with open(os.path.join(tasks_dir, f'{task}.json'), 'r') as f:
                task_info = json.load(f)
            category = task_info['Categories'][0].lower().replace(' ', '_')
            instruction = task_info['Definition']
            if isinstance(instruction, list):
                instruction = instruction[0]
            self.instructions.append(instruction)
            self.categories.append(category)
        self.unique_categories = sorted(set(self.categories))
        self.category_labels = np.array([self.unique_categories.index(category)
                                         for category in self.categories])

        # split into training, dev, and test sets stratifying by category
        np.random.seed(seed)
        train_idx, eval_idx = \
            train_test_split(np.arange(len(self.instructions)),
                             stratify=self.category_labels, train_size=0.8)
        dev_idx, test_idx = \
            train_test_split(eval_idx, stratify=self.category_labels[eval_idx],
                             train_size=0.5)

        # get indices for subset
        if subset == 'train':
            self.idx = train_idx
        elif subset == 'dev' or subset == 'eval':
            self.idx = dev_idx
        elif subset == 'test' or subset == 'predict':
            self.idx = test_idx

        self.len = len(self.idx)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        instruction = self.instructions[self.idx[idx]]
        inputs = self.tokenizer(instruction, max_length=self.max_length,
                                padding=True, truncation=True)
        inputs['label_ids'] = self.category_labels[self.idx[idx]]
        return inputs


def main(args, training_args):
    """ main script """

    # load tokenizer
    kwargs = {'cache_dir' : args.cache_dir}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **kwargs)

    # construct data subsets
    kwargs = {'max_length' : args.max_length, 'seed' : args.split_seed}
    train_dataset = TaskCategoryDataset(args.tasks_dir, args.task_file,
                                        tokenizer, subset='train', **kwargs)
    eval_dataset = TaskCategoryDataset(args.tasks_dir, args.task_file,
                                       tokenizer, subset='eval', **kwargs)
    test_dataset = TaskCategoryDataset(args.tasks_dir, args.task_file,
                                       tokenizer, subset='test', **kwargs)

    # load model, specifying number of outputs and multi-label classification
    kwargs = {
        'cache_dir' : args.cache_dir,
        'num_labels' : len(train_dataset.unique_categories),
    }
    model = \
        AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                           **kwargs)

    def compute_metrics(eval_pred):
        """ multi-class classification metrics function """
        labels, logits = eval_pred.label_ids, eval_pred.predictions
        preds = logits.argmax(axis=1)
        metrics = {
            'accuracy' : accuracy_score(labels, preds),
            'macro_f1' : f1_score(labels, preds, average='macro'),
            'micro_f1' : f1_score(labels, preds, average='micro'),
        }
        return metrics

    # set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        compute_metrics=compute_metrics,
    )

    # train
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # evaluate on dev set
    eval_results = trainer.predict(eval_dataset, metric_key_prefix='eval')
    metrics = eval_results.metrics
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)

    # evaluate on test set
    test_results = trainer.predict(test_dataset, metric_key_prefix='test')
    metrics = test_results.metrics
    trainer.log_metrics('test', metrics)
    trainer.save_metrics('test', metrics)

    trainer.save_model()
    trainer.save_state()


if __name__ == '__main__':

    # get command-line arguments
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    main(args, training_args)

