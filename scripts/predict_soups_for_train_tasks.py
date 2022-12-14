""" train pretrained language model as classifier for predicting greedy soup
    components from input text of instances for a set of tasks in the
    SuperNaturalInstructions dataset """

from argparse import ArgumentParser
from dataclasses import dataclass, field
import json
import os
from typing import Optional

import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
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

    resdir: Optional[str] = field(
        default='results/niv2-ts10-tr200-ev100-dev50/tk-instruct-base-experts'
                '/evaluate/greedy-soup-include-base-eval-task-train-tasks/all'
                '-instances-10000-steps',
        metadata={'help': 'path to greedy soup results for training tasks'},
    )
    tasks_dir: Optional[str] = field(
        default='data/tasks',
        metadata={'help': 'path to JSON info files for tasks'},
    )
    model_name: Optional[str] = field(
        default='roberta-base',
        metadata={'help': 'name of Huggingface model to fine-tune'},
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={'help': 'maximum length of input sequence'},
    )
    eval_task: Optional[int] = field(
        default=None,
        metadata={'help': 'index of task to use for evaluation'},
    )
    eval_category: Optional[int] = field(
        default=None,
        metadata={'help': 'index of category to use for evaluation'},
    )
    cache_dir: Optional[str] = field(
        default='/gscratch/ark/rahuln/.cache',
        metadata={'help': 'path to Huggingface cache directory'},
    )


class GreedySoupDataset(torch.utils.data.Dataset):
    """ dataset used for multi-label binary classification of greedy soup
        components from instances for a task in SuperNaturalInstructions """

    def __init__(self, resdir, tasks_dir, tokenizer, max_length=512,
        start_idx=0, end_idx=50, include_task=None, exclude_task=None,
        exclude_category=None, include_category=None):
        super(GreedySoupDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length

        # get list of tasks from results directory
        tasks = sorted(os.listdir(args.resdir))
        self.num_tasks = len(tasks)

        # get greedy soup components as labels for tasks
        task_labels = list()
        for task in tasks:
            with open(os.path.join(resdir, task, 'soup_info.json'), 'r') as f:
                soup_info = json.load(f)
            task_labels.append(soup_info['models'])
        binarizer = MultiLabelBinarizer()
        task_labels = binarizer.fit_transform(task_labels)
        self.task_labels = torch.from_numpy(task_labels).float().tolist()
        self.num_labels = task_labels.shape[1]

        # collect dev set instances and task category for each task
        self.instances = list()
        categories = set()
        for i, task in enumerate(tasks):
            with open(os.path.join(tasks_dir, f'{task}.json'), 'r') as f:
                task_info = json.load(f)
            task_instances = task_info['Instances']
            category = task_info['Categories'][0]
            categories.add(category)
            self.instances.extend([(i, inst['input'], category) for inst
                                   in task_instances[start_idx:end_idx]])
        categories = sorted(categories)

        # filter to include/exclude certain tasks
        if include_task is not None:
            self.instances = [(idx, inp) for idx, inp, cat in self.instances
                              if idx == include_task]
        elif exclude_task is not None:
            self.instances = [(idx, inp) for idx, inp, cat in self.instances
                              if idx != exclude_task]

        # filter to include/exclude certain categories
        if include_category is not None:
            self.instances = [(idx, inp) for idx, inp, cat in self.instances
                              if cat == categories[include_category]]
        elif exclude_category is not None:
            self.instances = [(idx, inp) for idx, inp, cat in self.instances
                              if cat != categories[exclude_category]]

        self.len = len(self.instances)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        task_idx, input_text = self.instances[idx]
        inputs = self.tokenizer(input_text, max_length=self.max_length,
                                padding=True, truncation=True)
        inputs['label_ids'] = self.task_labels[task_idx]
        return inputs


def main(args, training_args):
    """ main script """

    # load tokenizer
    kwargs = {'cache_dir' : args.cache_dir}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **kwargs)

    # load data subsets
    train_dataset = GreedySoupDataset(args.resdir, args.tasks_dir, tokenizer,
                                      max_length=args.max_length, start_idx=0,
                                      end_idx=40, exclude_task=args.eval_task,
                                      exclude_category=args.eval_category)
    eval_dataset = GreedySoupDataset(args.resdir, args.tasks_dir, tokenizer,
                                     max_length=args.max_length, start_idx=0,
                                     end_idx=40, exclude_task=args.eval_task,
                                     exclude_category=args.eval_category)
    test_dataset = GreedySoupDataset(args.resdir, args.tasks_dir, tokenizer,
                                     max_length=args.max_length, start_idx=0,
                                     end_idx=40, include_task=args.eval_task,
                                     include_category=args.eval_category)

    # load model, specifying number of outputs and multi-label classification
    kwargs['num_labels'] = train_dataset.num_labels
    kwargs['problem_type'] = 'multi_label_classification'
    model = \
        AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                           **kwargs)

    def compute_metrics(eval_pred):
        """ multi-label classification metrics function """
        labels, logits = eval_pred.label_ids, eval_pred.predictions
        preds = np.round(expit(logits))
        macro_f1 = f1_score(labels, preds, average='macro')
        micro_f1 = f1_score(labels, preds, average='micro')
        # exclude indices with no positive labels to avoid roc_auc_score error
        mask = np.nonzero(np.sum(labels, axis=0))[0]
        auroc = roc_auc_score(labels[:, mask], logits[:, mask])
        return {'macro_f1' : macro_f1, 'micro_f1' : micro_f1, 'auroc' : auroc}

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


if __name__ == '__main__':

    # get command-line arguments
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    main(args, training_args)

