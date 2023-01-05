#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from copy import deepcopy
from glob import glob
from itertools import product
from time import time
from typing import List, Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets.utils import set_progress_bar_enabled
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from ni_collator import DataCollatorForNI
from ni_trainer import NITrainer, DenserEvalCallback
from compute_metrics import compute_metrics, compute_grouped_metrics

import torch
from tqdm import tqdm

from run_s2s import DataTrainingArguments, ModelArguments, NITrainingArguments
from t5_output_ensemble import T5ForOutputEnsembling
from util import merge_state_dicts, send_to_device

set_progress_bar_enabled(False)
logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class SoupModelArguments(ModelArguments):
    """
    Extension of model arguments to handle construction of model soups
    """

    path_to_soup_components: str = field(
        default=None,
        metadata={"help": "Path to directory containing trained models to use as soup components."}
    )
    max_soup_size: int = field(
        default=10,
        metadata={"help": "Maximum number of components for the model soup."}
    )
    include_base_model: bool = field(
        default=False,
        metadata={"help": "Include base model as possible soup component."}
    )
    start_with_base_model: bool = field(
        default=False,
        metadata={"help": "Use base model as initial soup component."}
    )
    output_ensemble: bool = field(
        default=False,
        metadata={"help": "Use output ensemble instead of parameter average."}
    )
    include_models: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Specific paths to other models to include as soup components."}
    )
    param_groups: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Regex patterns for parameter names to form parameter groups."}
    )

@dataclass
class SoupDataArguments(DataTrainingArguments):
    """
    Extension of data arguments to handle construction of model soups
    """

    use_train_as_dev: Optional[bool] = field(
        default=False,
        metadata={"help": "use training set as dev set"}
    )
    use_test_as_dev: Optional[bool] = field(
        default=False,
        metadata={"help": "use test set as dev set"}
    )
    eval_instance_ids_file: Optional[str] = field(
        default=None,
        metadata={"help": "Filename with list of instance IDs to keep in dev set."}
    )
    test_task: Optional[str] = field(
        default=None,
        metadata={"help": "Name of task to use for filtering test set."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    start = time()

    parser = HfArgumentParser((SoupModelArguments, SoupDataArguments, NITrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the NaturalInstructions dataset
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=data_args.data_dir, 
        task_dir=data_args.task_dir, 
        cache_dir=model_args.cache_dir,
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        use_dev=data_args.use_dev,
        relative_scales_file=data_args.relative_scales_file,
        reduction_factor=data_args.reduction_factor,
        num_dev=data_args.num_dev
    )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model_args.output_ensemble:
        model = T5ForOutputEnsembling([model],
                                      base_model=model_args.model_name_or_path,
                                      cache_dir=model_args.cache_dir)
        model.remove_model()

    # load state_dicts of all models that could be components in the soup
    files = sorted(glob(os.path.join(model_args.path_to_soup_components, '**',
                                     'pytorch_model.bin'), recursive=True))
    if model_args.include_models is not None:
        if len(model_args.include_models) == 1:
            include_models = model_args.include_models[0].split(',')
        else:
            include_models = model_args.include_models
        files.extend(include_models)

    state_dicts = list()
    for fname in tqdm(files, desc='loading state_dicts'):
        if model_args.output_ensemble:
            component = AutoModelForSeq2SeqLM.from_pretrained(
                fname, config=config, cache_dir=model_args.cache_dir)
            state_dicts.append(component)
        else:
            state_dicts.append(torch.load(fname, map_location='cpu'))

    # include base model as potential soup component
    if model_args.include_base_model or model_args.start_with_base_model:
        logger.info("Including base model as potential soup component")
        if model_args.output_ensemble:
            component = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path, config=config,
                cache_dir=model_args.cache_dir)
            state_dicts.insert(0, component)
        else:
            state_dicts.insert(0, send_to_device(model.state_dict(), 'cpu'))
        files.insert(0, model_args.model_name_or_path)

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if "train" not in raw_datasets:
        raise ValueError("train dataset required")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if "validation" not in raw_datasets:
        raise ValueError("validation dataset required")
    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if "test" not in raw_datasets:
        raise ValueError("test dataset required")
    predict_dataset = raw_datasets["test"]
    if data_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # use test set as validation set
    if data_args.use_test_as_dev:
        logger.info("Using test set as validation set")
        eval_dataset = predict_dataset

    # use training set as validation set
    if data_args.use_train_as_dev:
        logger.info("Using training set as validation set")
        eval_dataset = train_dataset

    # restrict instances in eval_dataset to those with IDs in specified file
    if data_args.eval_instance_ids_file is not None:
        logger.info(f'Restricting eval_dataset to instances with IDs in: {data_args.eval_instance_ids_file}')
        with open(data_args.eval_instance_ids_file, 'r') as f:
            keep_ids = set([line.strip() for line in f.readlines()])

        def id_filter(example):
            return example['id'] in keep_ids

        eval_dataset = eval_dataset.filter(id_filter)

    # restrict test set instances to those from specified task
    if data_args.test_task is not None:
        logger.info(f'Restricting predict_dataset to task: {data_args.test_task}')

        def task_filter(example):
            return example['Task'] == data_args.test_task

        predict_dataset = predict_dataset.filter(task_filter)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForNI(
        tokenizer,
        model=model,
        padding="max_length" if data_args.pad_to_max_length else "longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_task_definition=data_args.add_task_definition,
        num_pos_examples=data_args.num_pos_examples,
        num_neg_examples=data_args.num_neg_examples,
        add_explanation=data_args.add_explanation,
        tk_instruct=data_args.tk_instruct
    )
    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False 

    # Metric

    def compute_ni_metrics(dataset, preds, save_prefix=None):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [e["Instance"]["output"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=dataset["Task"])
        result.update(result_per_task)
        categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Definition": example["Definition"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    # Initialize our Trainer
    trainer = NITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_ni_metrics if training_args.predict_with_generate else None,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    all_metrics = {"run_name": training_args.run_name}

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    # find single best model to start soup with
    soup_info = {"models" : list(), "eval_rougeL" : list()}
    best_metric, best_idx = -np.inf, 0
    if model_args.start_with_base_model:
        if model_args.output_ensemble:
            state_dicts[0] = state_dicts[0].cuda()
            model.add_model(state_dicts[0])
            model.send_to_device('cuda:0')
        logger.info("Using base model as initial soup component")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        best_metric, best_idx = metrics["eval_rougeL"], 0
    else:
        logger.info("Finding best initial model")
        for idx, state_dict in enumerate(state_dicts):
            if model_args.output_ensemble:
                state_dict = state_dict.cuda()
                model.add_model(state_dict)
                model.send_to_device('cuda:0')
            else:
                model.load_state_dict(state_dict)
            metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
            if metrics["eval_rougeL"] > best_metric:
                best_metric, best_idx = metrics["eval_rougeL"], idx
            if model_args.output_ensemble:
                component = model.remove_model()
                component = component.cpu()
    logger.info(f"best eval_rougeL: {best_metric:.2f}, index: {best_idx}")
    soup_info["models"].append(files[best_idx])
    soup_info["eval_rougeL"].append(best_metric)
    if model_args.output_ensemble:
        state_dicts[best_idx] = state_dicts[best_idx].cuda()
        model.add_model(state_dicts[best_idx])
    else:
        model.load_state_dict(state_dicts[best_idx])

    # get patterns for parameter groups
    if model_args.param_groups is not None:
        if model_args.output_ensemble:
            raise ValueError('parameter groups are currently not supported '
                             'when using an output ensemble')
        if len(model_args.param_groups) == 1:
            param_groups = model_args.param_groups[0].split(',')
        else:
            param_groups = model_args.param_groups
        logger.info(f"Using parameter groups: {param_groups}")
    else:
        param_groups = [".*"]

    # for maximum number of iterations, add a model to the soup based on performance
    if not model_args.output_ensemble:
        curr_state_dict = deepcopy(model.state_dict())
    for it in range(model_args.max_soup_size - 1):
        logger.info(f"Running evaluation {it+2} / {model_args.max_soup_size}")
        prev_best_metric = best_metric
        for idx, (state_dict, pattern) in enumerate(product(state_dicts, param_groups)):
            if model_args.output_ensemble:
                state_dict = state_dict.cuda()
                model.add_model(state_dict)
                model.send_to_device('cuda:0')
            else:
                new_state_dict = merge_state_dicts(curr_state_dict, state_dict, num_averaged=it+1, pattern=pattern)
                model.load_state_dict(new_state_dict)
            metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
            if metrics["eval_rougeL"] > best_metric:
                best_metric, best_idx = metrics["eval_rougeL"], idx
            if model_args.output_ensemble:
                component = model.remove_model()
                component = component.cpu()
        logger.info(f"best eval_rougeL: {best_metric:.2f}, index: {best_idx}")
        if prev_best_metric == best_metric:
            logger.info("metric did not improve, exiting loop")
            break
        if model_args.output_ensemble:
            state_dicts[best_idx] = state_dicts[best_idx].cuda()
            model.add_model(state_dicts[best_idx])
            soup_info["models"].append(files[best_idx])
        else:
            best_state_dict = state_dicts[best_idx // len(param_groups)]
            best_pattern = param_groups[best_idx % len(param_groups)]
            curr_state_dict = merge_state_dicts(curr_state_dict, best_state_dict, num_averaged=it+1, pattern=best_pattern)
            soup_info["models"].append((files[best_idx // len(param_groups)], best_pattern))
        soup_info["eval_rougeL"].append(best_metric)

    # load state_dict for best soup into model
    if not model_args.output_ensemble:
        model.load_state_dict(curr_state_dict)

    # run final evaluation on dev set
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    all_metrics.update(metrics)

    # run final evaluation on test set
    logger.info("*** Predict ***")
    predict_results = trainer.predict(
        predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
    )
    metrics = predict_results.metrics
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
    trainer.log(metrics)
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    all_metrics.update(metrics)

    # write all metrics to file
    with open(os.path.join(training_args.output_dir, "metrics.json"), "w") as fout:
        json.dump(all_metrics, fout, indent=4)

    # write soup info to file
    elapsed = time() - start
    soup_info['soup_runtime'] = elapsed
    with open(os.path.join(training_args.output_dir, "soup_info.json"), "w") as fout:
        json.dump(soup_info, fout, indent=4)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
