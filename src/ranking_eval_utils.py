""" utilities for evaluation using ranking classification, adapted from
    t0/model.py and t0/data_collator.py in the T0 codebase:

    https://github.com/bigscience-workshop/t-zero """

from dataclasses import dataclass
from itertools import chain
import logging
from typing import Optional, Union

import torch
from torch import nn
from transformers import (
    AutoModelForSeq2SeqLM,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    PreTrainedTokenizerBase,
)
from transformers.file_utils import PaddingStrategy

logger = logging.getLogger(__name__)


class ModelBase(nn.Module):
    def forward(self, batch):
        raise NotImplementedError

    @staticmethod
    def from_config(config, **kwargs) -> "ModelBase":
        task_mapping = [
            (MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, EncoderDecoderModel),
        ]
        config_name = config.__class__
        for transformer_model_mapping, model in task_mapping:
            transformer_model_name = transformer_model_mapping.get(config_name, None)
            if transformer_model_name is not None:
                return model(config=config, **kwargs)

        raise NotImplementedError


class EncoderDecoderModel(ModelBase):
    def __init__(self, model=None, model_name_or_path=None, cache_dir=None,
        **kwargs):
        """

        Args:
            config:
            model_name_or_path:
        """
        super(EncoderDecoderModel, self).__init__()
        logger.info("Building EncoderDecoderModel")
        if model is not None:
            self._model = model
        elif model_name_or_path:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                cache_dir=cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            self._model = AutoModelForSeq2SeqLM.from_config(config)

    def forward(self, batch) -> torch.Tensor:
        model_inputs = {
            k: batch[k]
            for k in ["input_ids", "attention_mask", "labels"]
        }
        logits = self._model(**model_inputs).logits
        masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
        seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0),
                                         -1)  # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
        predictions = seq_log_prob.argmax(dim=-1)
        return predictions


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [
                {
                    k: v[i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id]*(max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0]*(max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch

