# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
# from fairseq.modules.moe_layer import MoELayer
from omegaconf import II


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("moe_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class MoECrossEntropyCriterion(FairseqCriterion):
    moe_logging_keys = [
        "overflow_expert1",        # average % of overflowed tokens from 1st expert
        # "overflow_expert2",        # average % of overflowed tokens from 2nd expert
        "entropy_gating",          # average entropy of the gating distribution
        "expert1_balance_top",     # average cumulative % of tokens processed by the most used 20% 1st experts
        "expert1_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 1st experts
        "unused_expert1_count",    # average number of 1st experts which process no tokens
        # "expert2_balance_top",     # average cumulative % of tokens processed by the most used 20% 2nd experts
        # "expert2_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 2nd experts
        # "unused_expert2_count",    # average number of 2nd experts which process no tokens
        "all_to_all_cpu_time_ms",  # CPU time spent in all to all calls in milliseconds
        "all_to_all_cuda_time_ms", # CUDA ttime spent in all to all calls in milliseconds
    ]
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, inner_loss, moe_loss, moe_metadata, sample_size, logging_output = self.compute_loss(model, sample, reduce=reduce)

        logging_output["loss"] = loss.data
        logging_output["moe_loss"] = moe_loss
        # logging_output["moe_loss"] = moe_loss.data
        logging_output.update(moe_metadata)

        return loss, sample_size, logging_output

    def compute_inner_loss(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        nll_loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        logging_output = {
            "inner_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return net_output, nll_loss, sample_size, logging_output

    def compute_loss(self, model, sample, reduce=True):
        net_output, inner_loss, sample_size, logging_output = self.compute_inner_loss(model, sample)
        gate_loss = 0.0
        gate_count = 0
        for name, module in model.named_modules():
            if callable(getattr(module, "calc_last_bloss", None)):
                gate_loss += module.calc_last_bloss()
                gate_count += 1
        # TODO My read is that this is how they're passing the loss back, but I'm calling the func directly
        # double check that I don't need this
        # for l_aux in net_output[1]["l_aux"]:
        #     if l_aux is not None:
        #         gate_loss += l_aux
        #         gate_count += 1
        # if self.gate_loss_combine_method == "average":
        #     gate_loss = gate_loss / gate_count
        # if self.gate_loss_transform == "neg_log":
        #     gate_loss = - torch.log(gate_loss)
        # gate_loss = sample_size * gate_loss
        loss = inner_loss + gate_loss
        return loss, inner_loss, gate_loss, self.get_moe_metadata(model), sample_size, logging_output

    def get_moe_metadata(self, model):
        moe_logging_output = {}
        for key in MoECrossEntropyCriterion.moe_logging_keys:
            total_val = 0
            count = 0
            for _, module in model.named_modules():
                # if isinstance(module, MoELayer):
                # NOTE: I have a circular import somewhere if I try to import MoELayer. 
                # Haven't figure out how to resolve it yet
                if getattr(module, "moe_metadata", None):
                    total_val += module.moe_metadata[key] if key in module.moe_metadata else 0
                    count += 1
            moe_logging_output[key] = total_val / count
        moe_logging_output["batch_count"] = 1
        return moe_logging_output

    @staticmethod
    def reduce_moe_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        moe_loss_sum = sum(log.get("moe_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "moe_gate_loss", moe_loss_sum / sample_size, sample_size, round=8
        )
        batch_count = sum(log.get("batch_count", 0) for log in logging_outputs)
        for key in MoECrossEntropyCriterion.moe_logging_keys:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(
                key, val / batch_count, batch_count, round=3
            )

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        MoECrossEntropyCriterion.reduce_moe_metrics(logging_outputs)

        loss_sum = sum(log.get("inner_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "inner_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["inner_loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True