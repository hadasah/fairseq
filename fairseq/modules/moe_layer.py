# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

from fairseq import utils
from fairseq.distributed import utils as distributed_utils
# from fairseq.modules import FairseqDropout, LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.layer_norm import LayerNorm
from fairseq.modules.multihead_attention import MultiheadAttention
from torch import Tensor
import torch.nn.init as init


LOGGING_SAMPLE_FRACTION = 0.2

class MoELayer(nn.Module):

    def __init__(self, args, shared_moe_experts=None):
        super().__init__()
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        # expert_centroids = torch.empty(self.num_workers, args.decoder_embed_dim)
        # torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        # self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        # self.register_parameter("expert_centroids", torch.nn.Linear(args.decoder_embed_dim, self.num_workers, bias=False))
        self.expert_centroids = torch.nn.Linear(args.decoder_embed_dim, self.num_workers, bias=False)
        # with torch.no_grad():
        #     init.xavier_uniform_(self.expert_centroids.weight, gain=init.calculate_gain('relu')/(self.num_workers))
        self.expert_network = (
            shared_moe_experts if shared_moe_experts
            else nn.Sequential(*([MoESublayer(args) for _ in range(args.base_sublayers)]))
        )
        self.expert_id = distributed_utils.get_data_parallel_rank()
        self.shuffle = args.base_shuffle
        self.bloss_type = args.moe_bloss_type
        self.bloss_weight = args.moe_bloss_weight
        self.use_fp32_gating = args.moe_use_fp32_gating
        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        # 
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        # input_features is s x b x e
        # features is now sb x e (which I just refer to as s x e afterwards)
        transposed_features = input_features.transpose(0, 1)
        features = transposed_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2All.apply(features[shuffle_sort])

        # Compute which token goes to which expert
        routing_strategy = 'greedy'
        sort_by_expert, input_splits, output_splits, self.moe_metadata = self.assignment(features, strategy=routing_strategy)
        # Swap these tokens for the right ones for our expert
        routed_features = All2All.apply(features[sort_by_expert], output_splits, input_splits)

        if routed_features.size(0) > 0:
            routed_features = self.expert_network(routed_features)
        # Return to original worker and ordering
        result = All2All.apply(routed_features, input_splits, output_splits)[self.inverse_sort(sort_by_expert)]

        if self.shuffle and is_training:
            # Undo shuffling
            result = All2All.apply(result)[self.inverse_sort(shuffle_sort)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(transposed_features.size()).transpose(0, 1), None, None

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    # Assigns each token to the top k experts
    # NOTE: this ONLY works for k=1 right now b/c of self.last_mask assignment
    # features is s x e
    def assignment(self, features, strategy='greedy', k=1):
        def entropy(probs): #, logits):
            logits = torch.distributions.utils.probs_to_logits(probs)
            p_log_p = probs * logits
            return -p_log_p.sum(-1)

        metadata = {}
        # Compute similarity of each token to each expert, for routing
        # features is s x e , expert_centroids is e x h 
        logits = self.expert_centroids(features)
        # logits = features.matmul(self.expert_centroids.transpose(0, 1))
        # logits has dims s x e
        if self.use_fp32_gating:
            orig_dtype = logits.dtype
            logits = logits.float()
        # TODO @margsli this is only argmax, should the other sampling strategies be implemented?
        self.last_p = torch.softmax(logits, -1)
        metadata["entropy_gating"] = entropy(probs=self.last_p).mean().detach()

        if strategy == 'greedy':
            # get a flattened list of the workers to send each token to
            # logits is [sxb,e], token_to_workers is [sxb], with the worker # for each token
            idx = torch.topk(logits, dim=1, k=k, largest=True).indices
            self.last_mask = F.one_hot(idx[:, 0], num_classes=self.num_workers)

            # log expert usage
            num_tokens = logits.shape[0]
            expert1_hist = 100 * torch.histc((idx.squeeze() + 1), bins=self.num_workers, min=1, max=self.num_workers) / num_tokens
            metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
            expert1_hist = torch.sort(expert1_hist, dim=0, descending=True).values + torch.finfo(torch.float32).tiny

            # log whether the routing is balanced (top and bottom)
            sample_count = max(math.ceil(self.num_workers * LOGGING_SAMPLE_FRACTION), 1)
            metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
            metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

            token_to_workers = idx.view(-1)
            # token_to_workers is worker # for each token, sorted, sort_ordering gives original index
            token_to_workers, sort_ordering = torch.sort(token_to_workers)
            # this works because 2n and 2n+1 are the same token, diff experts
            worker2token = sort_ordering // k

            # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
            output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=logits.device)
            workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
            output_splits[workers] = counts
            # Tell other workers how many tokens to expect from us
            input_splits = All2All.apply(output_splits)
            return worker2token, input_splits.tolist(), output_splits.tolist(), metadata
        
        else:
            raise Error("strategy not implemented")

    def calc_last_bloss(self):
        if not self.training: return 0.0

        # GShard basline uses a group size of 1024
        # TODO @margsli check these values
        group_size = 1024.0
        tokens_per_batch = 128*256
        gpus = 8
        micro_batches = 1
         # since the TF baseline uses modelparallelism, the losses of each group as summed and not averaged
         # this is a constant offset to bring the pytorch loss-weight in-line with the tensorflow one
         # this was experimentally verified: with a loss-weight of 0.01 the ratio of gate loss to total loss
         # is about 500 at the beginning of training
        tf_constant_scale = (tokens_per_batch/group_size)*gpus*micro_batches
        if self.last_p is not None:
            if self.bloss_type == 'mean-diff':
                frac = self.last_mask.float().mean(0)
                p = self.last_p.float().mean(0)
                frac -= 1./self.num_workers
                frac = torch.abs(frac)
            elif self.bloss_type == 'mean':
                frac = self.last_mask.float().mean(0)
                p = self.last_p.float().mean(0)
            else:
                print(self.bloss_type)
                raise NotImplementedError('Loss type not implemented')

            # we do sum reduction instead of mean reduction; in fairseq the loss is normalized later
            loss = tf_constant_scale*(frac*p).mean()*(self.num_workers**2)*self.bloss_weight

            return loss
        else:
            return 0.0


class MoESublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.norm = LayerNorm(args.decoder_embed_dim, export=False)
        self.ff1 = torch.nn.Linear(args.decoder_embed_dim, args.decoder_ffn_embed_dim)
        self.norm2 = LayerNorm(args.decoder_ffn_embed_dim, export=False)
        self.ff2 = torch.nn.Linear(args.decoder_ffn_embed_dim, args.decoder_embed_dim)
        self.norm3 = LayerNorm(args.decoder_embed_dim, export=False)
        self.norm4 = LayerNorm(args.decoder_embed_dim, export=False)
        self.ff2.weight.data.zero_()

    def forward(self, xs):

        # set norm correctly
        # if self.norm2:
        # return xs + self.ff2(self.activation_fn(self.norm2(self.ff1(self.norm(xs)))))
        return xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))
        # return self.norm4(xs + self.norm3(self.ff2(self.norm2(self.activation_fn(self.ff1(self.norm(xs)))))))
        # return xs + self.norm(self.ff2(self.activation_fn(self.ff1(xs))))


# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        torch.distributed.all_to_all_single(ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        torch.distributed.all_to_all_single(result, grad_output,
                                            output_split_sizes=ctx.input_splits, input_split_sizes=ctx.output_splits)
        return result, None, None

class MoETransformerDecoderLayerBase(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, 
        shared_moe_layer=None, shared_moe_experts=None, 
    ):
        super().__init__()
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        # TODO @margsli Does it matter if I use the simpler relu activation code above or use this?
        # If so, I can instantiate the MoE with this activation function directly
        # self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        # activation_dropout_p = cfg.activation_dropout
        # if activation_dropout_p == 0:
        #     # for backwards compatibility with models that use cfg.relu_dropout
        #     activation_dropout_p = cfg.relu_dropout or 0
        # self.activation_dropout_module = FairseqDropout(
        #     float(activation_dropout_p), module_name=self.__class__.__name__
        # )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        # self.fc1 = self.build_fc1(
        #     self.embed_dim,
        #     cfg.decoder.ffn_embed_dim,
        #     self.quant_noise,
        #     self.quant_noise_block_size,
        # )
        # self.fc2 = self.build_fc2(
        #     cfg.decoder.ffn_embed_dim,
        #     self.embed_dim,
        #     self.quant_noise,
        #     self.quant_noise_block_size,
        # )
        
        # MoE layer is router -> LayerNorm -> FF1 -> Relu -> FF2
        self.moe = shared_moe_layer if shared_moe_layer else MoELayer(cfg, shared_moe_experts=shared_moe_experts)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False

    # def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
    #     return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    # def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
    #     return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        # x = self.activation_fn(self.fc1(x))
        # x = self.activation_dropout_module(x)
        # x = self.fc2(x)
        x, _, _ = self.moe(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

# backward compatible with the legacy argparse format
class MoETransformerDecoderLayer(MoETransformerDecoderLayerBase):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, 
        shared_moe_layer=None, shared_moe_experts=None, 
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
        )
