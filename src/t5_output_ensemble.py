""" class modified from T5ForConditionalGeneration to allow for output
    ensembling of multiple T5 models at inference time """

from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import T5PreTrainedModel, T5ForConditionalGeneration
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)


class T5ForOutputEnsembling(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, models, base_model='t5-base', cache_dir=None):
        super().__init__(models[0].config)

        self.base_model_name = base_model
        self.cache_dir = cache_dir

        # declarations below copied from T5ForConditionalGeneration.__init__
        if len(models) > 0:
            self.model_dim = models[0].config.d_model
            self.encoder = models[0].encoder
            self.decoder = models[0].decoder
            self.lm_head = models[0].lm_head

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.models = models

    def send_to_device(self, device):
        for model in self.models:
            model = model.to(device)

    def get_encoder(self):

        def fn(input_ids, return_dict=True, **encoder_kwargs):
            encoder_outputs = list()
            for model in self.models:
                model_encoder_outputs = \
                    model.encoder(input_ids, return_dict=True, **encoder_kwargs)
                encoder_outputs.append(model_encoder_outputs)
            return encoder_outputs

        return fn

    def get_decoder(self):
        return self.decoder

    def add_model(self, model):
        if not isinstance(model, T5ForConditionalGeneration):
            self._add_model_from_state_dict(model)
        self.models.append(model)
        if len(self.models) == 1:
            self.model_dim = self.models[0].config.d_model
            self.encoder = self.models[0].encoder
            self.decoder = self.models[0].decoder
            self.lm_head = self.models[0].lm_head

    def _add_model_from_state_dict(self, state_dict):
        model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, cache_dir=self.cache_dir)
        model.load_state_dict(state_dict)
        self.add_model(model)

    def remove_model(self):
        return self.models.pop()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """ modified version of forward function that performs forward pass
            on all models in list of models, averaging output logits """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = list()
            for model in self.models:
                model_encoder_outputs = model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                encoder_outputs.append(model_encoder_outputs)
        elif return_dict and not \
            all(isinstance(model_encoder_outputs, BaseModelOutput) for model_encoder_outputs in encoder_outputs):
            for i, model_encoder_outputs in enumerate(encoder_outputs):
                encoder_outputs[i] = BaseModelOutput(
                    last_hidden_state=model_encoder_outputs[0],
                    hidden_states=model_encoder_outputs[1] if len(model_encoder_outputs) > 1 else None,
                    attentions=model_encoder_outputs[2] if len(model_encoder_outputs) > 2 else None,
                )

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)
        #     hidden_states = hidden_states.to(self.decoder.first_device)
        #     if decoder_input_ids is not None:
        #         decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(self.decoder.first_device)
        #     if decoder_attention_mask is not None:
        #         decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = list()
        for i, model in enumerate(self.models):
            model_past_key_values = past_key_values[i] if past_key_values is not None else None
            model_decoder_inputs_embeds = decoder_inputs_embeds[i] if decoder_inputs_embeds is not None else None
            model_decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=model_decoder_inputs_embeds,
                past_key_values=model_past_key_values,
                encoder_hidden_states=encoder_outputs[i][0],
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            decoder_outputs.append(model_decoder_outputs)

        # # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.encoder.first_device)
        #     self.lm_head = self.lm_head.to(self.encoder.first_device)
        #     sequence_output = sequence_output.to(self.lm_head.weight.device)

        # if self.config.tie_word_embeddings:
        #     # Rescale output before projecting on vocab
        #     # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        #     sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = list()
        for i, model_decoder_outputs in enumerate(decoder_outputs):
            sequence_output = model_decoder_outputs[0]
            model_lm_logits = self.models[i].lm_head(sequence_output)
            lm_logits.append(model_lm_logits)

        # average logits across models
        lm_logits = torch.mean(torch.stack(lm_logits), dim=0)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[0][1:] + encoder_outputs[0]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=[elem.past_key_values for elem in decoder_outputs],
            decoder_hidden_states=[elem.hidden_states for elem in decoder_outputs],
            decoder_attentions=[elem.attentions for elem in decoder_outputs],
            cross_attentions=[elem.cross_attentions for elem in decoder_outputs],
            encoder_last_hidden_state=[elem.last_hidden_state for elem in encoder_outputs],
            encoder_hidden_states=[elem.hidden_states for elem in encoder_outputs],
            encoder_attentions=[elem.attentions for elem in encoder_outputs],
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

