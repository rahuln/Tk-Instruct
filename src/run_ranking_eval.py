""" evaluate model using ranking classification, adapted from
    evaluation/run_eval.py in the T0 codebase:

    https://github.com/bigscience-workshop/t-zero """

import argparse
from itertools import chain
import logging
import os
import random
import json

import datasets
import torch
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
)

from ni_collator import DataCollatorForNI
from ranking_eval_utils import DataCollatorForMultipleChoice, EncoderDecoderModel

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce main evaluation in T0.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="The directory for saving the NaturalInstructions train/dev/test splits.",
        required=True,
    )
    parser.add_argument(
        "--task_dir",
        type=str,
        default=None,
        help="The directory for saving the NaturalInstructions tasks json files.",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--max_num_instances_per_task",
        type=int,
        default=None,
        help="The maximum number of instances we will consider for each training task.",
    )
    parser.add_argument(
        "--max_num_instances_per_eval_task",
        type=int,
        default=500,
        help="The maximum number of instances we will consider for each validation/test task.",
    )
    parser.add_argument(
        "--use_dev",
        type=bool,
        default=False,
        help="split test instances into dev and test sets, ensure that they are left out of training set",
    )
    parser.add_argument(
        "--num_dev",
        type=int,
        default=50,
        help="number of dev set examples to use per task",
    )
    parser.add_argument(
        "--train_on_dev",
        type=bool,
        default=False,
        help="train on dev set instead of training set",
    )
    parser.add_argument(
        "--relative_scales_file",
        type=str,
        default=None,
        help="path to file with relative scaling factors for tasks for upsampling tasks to be the same size.",
    )
    parser.add_argument(
        "--reduction_factor",
        type=float,
        default=None,
        help="reduction factor for downsampling the number of instances per task",
    )
    parser.add_argument(
        "--add_task_name",
        type=bool,
        default=False,
        help="whether to preappend task name before the task input.",
    )
    parser.add_argument(
        "--add_task_definition",
        type=bool,
        default=True,
        help="whether to preappend task definition before the task input.",
    )
    parser.add_argument(
        "--num_pos_examples",
        type=int,
        default=0,
        help="number of in-context positive examples.",
    )
    parser.add_argument(
        "--num_neg_examples",
        type=int,
        default=0,
        help="number of in-context negative examples.",
    )
    parser.add_argument(
        "--add_explanation",
        type=bool,
        default=False,
        help="whether to add explanation for both the postive examples and negtive examples.",
    )
    parser.add_argument(
        "--tk_instruct",
        type=bool,
        default=False,
        help="tk_instruct will train a model combining all valid instruction encodings. This will overwrite the other settings about instruction encoding.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="Target max length. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models. The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use half-precision for model parameters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation.",
    )
    args = parser.parse_args()

    return args


def eval_ranking(args, model, tokenizer, eval_dataset, ex_answer_choices,
    batch_size=8, cache_dir=None, use_fp16=False, device='cuda'):
    """ run evaluation with ranking classification, return accuracy """

    # construct warpper for model to perform ranking classification
    model = EncoderDecoderModel(model=model, cache_dir=cache_dir)
    model.to(device)

    # load NaturalInstructions data collator
    data_collator = DataCollatorForNI(
        tokenizer,
        model=model,
        padding="max_length" if args.pad_to_max_length else "longest",
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8 if use_fp16 else None,
        add_task_name=args.add_task_name,
        add_task_definition=args.add_task_definition,
        num_pos_examples=args.num_pos_examples,
        num_neg_examples=args.num_neg_examples,
        add_explanation=args.add_explanation,
        tk_instruct=args.tk_instruct,
        text_only=True
    )

    # apply data collator to dataset to convert inputs to prompt format
    inputs = [elem['Instance']['input'] for elem in eval_dataset]
    eval_dataset = Dataset.from_dict(data_collator(eval_dataset))
    labels = eval_dataset['labels']

    # get column names from dataset
    column_names = eval_dataset.column_names

    def preprocess_function(examples):
        bs = len(examples[column_names[0]])
    
        input_texts = []
        target_texts = []
        answer_choices_texts = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in column_names
            }
            assert ex["labels"] in ex_answer_choices
            input_texts.append(ex["inputs"])
            target_texts.append(ex["labels"])
            answer_choices_texts.append(ex_answer_choices)
    
        tokenized_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_source_length,
            truncation=True,
        )
        tokenized_targets = [
            tokenizer(
                ans_choi,
                # padding is on the right here.
                padding=False,
                max_length=args.max_target_length,
                truncation=True,
            )
            for ans_choi in answer_choices_texts
        ]
    
        features = {
            k: [
                [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in tokenized_inputs.items()
        }
    
        features["labels"] = [
            tokenized_targets[idx]["input_ids"]
            for idx in range(bs)
        ]
        features["labels_attention_mask"] = [
            tokenized_targets[idx]["attention_mask"]
            for idx in range(bs)
        ]
        features["targets"] = [
            answer_choices_texts[idx].index(t)
            for idx, t in enumerate(target_texts)
        ]
    
        return features

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False
    eval_dataset = eval_dataset.map(
        preprocess_function, batched=True, remove_columns=column_names
    )

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if use_fp16 else None)
        )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    # Metrics
    metric = load_metric("accuracy")

    # Eval!
    total_batch_size = batch_size
    progress_bar = tqdm(range(len(eval_dataloader)))
    model.eval()
    predicted_labels = list()
    for batch in eval_dataloader:
        batch = {k : v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            predictions = model(batch)

        metric.add_batch(
            predictions=predictions,
            references=batch["targets"],
        )
        predicted_labels.extend([ex_answer_choices[p] for p in predictions])

        progress_bar.update(1)

    eval_metric = metric.compute()
    print(f"Result: {eval_metric}")

    results = {
        "accuracy": eval_metric["accuracy"]
    }

    return results


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Setup logging
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    # load tokenizer, set pad token
    tokenizer = \
        AutoTokenizer.from_pretrained(args.model_name_or_path,
                                      use_fast=not args.use_slow_tokenizer,
                                      cache_dir=args.cache_dir)

    if tokenizer.pad_token is None:
        for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
            if token is not None:
                tokenizer.pad_token = token
        if tokenizer.pad_token is None:
            raise ValueError("Please define a pad token id.")

    # load NaturalInstructions dataset
    raw_datasets = load_dataset(
        "src/ni_dataset.py", 
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        cache_dir=args.cache_dir,
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task,
        use_dev=args.use_dev,
        relative_scales_file=args.relative_scales_file,
        reduction_factor=args.reduction_factor,
        num_dev=args.num_dev,
        train_on_dev=args.train_on_dev
    )

    # get possible answer choices from training dataset
    all_outputs = list(chain(*[ex["Instance"]["output"]
                               for ex in raw_datasets["train"]]))
    ex_answer_choices = sorted(set(all_outputs))

    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    # run ranking classification evaluation, collect results
    results = eval_ranking(args, model, tokenizer, raw_datasets["test"],
                           ex_answer_choices, cache_dir=args.cache_dir,
                           use_fp16=args.use_fp16)

    # Handle the output directory creation
    os.makedirs(args.output_dir, exist_ok=True)

    # save results to file
    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
