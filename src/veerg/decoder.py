from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_pubmed_gpt(path: str, gradient_checkpointing: bool = True):
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config.from_pretrained(path, local_files_only=True)
    config.add_cross_attention = True
    config.is_decoder = True
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    model = GPT2LMHeadModel.from_pretrained(path, config=config, local_files_only=True)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return tokenizer, model


def pack_phrases(tokenizer, phrases: Sequence[str], max_length: int, device):
    if not phrases or max_length < 1:
        raise ValueError("phrases and max_length must be non-empty")
    rows, label_rows = [], []
    truncated = 0
    for phrase in phrases:
        body = tokenizer.encode(str(phrase), add_special_tokens=False)
        truncated += int(len(body) > max_length)
        target = [*body[:max_length], tokenizer.eos_token_id]
        rows.append([tokenizer.bos_token_id, *target])
        label_rows.append([-100, *target])
    width = max(map(len, rows))
    ids = torch.full(
        (len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.full_like(ids, -100)
    for index, (row, label_row) in enumerate(zip(rows, label_rows)):
        ids[index, : len(row)] = torch.tensor(row, device=device)
        attention[index, : len(row)] = True
        labels[index, : len(label_row)] = torch.tensor(label_row, device=device)
    return ids, attention, labels, truncated


def token_ce_sums(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = logits[:, :-1].float()
    targets = labels[:, 1:]
    losses = F.cross_entropy(
        scores.reshape(-1, scores.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(targets)
    valid = targets.ne(-100)
    return (losses * valid).sum(-1), valid.sum(-1)


def token_level_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    loss_sums, token_counts = token_ce_sums(logits, labels)
    return loss_sums.sum() / token_counts.sum().clamp_min(1)


class RegionalDecoder(nn.Module):
    """One shared GPT-2 decoder reused for every selected anatomical region."""

    def __init__(
        self,
        tokenizer,
        lm: nn.Module,
        max_length: int = 64,
        num_beams: int = 3,
        length_penalty: float = 1.1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.lm = lm
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("decoder target lengths must be positive")
        self.num_beams = int(num_beams)
        self.length_penalty = float(length_penalty)

    def _validate(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> None:
        if memory.ndim != 3 or memory.shape[1] < 1:
            raise ValueError("regional memory must have shape [N, S>=1, H]")
        if memory.shape[-1] != int(self.lm.config.n_embd):
            raise ValueError("memory hidden dimension does not match GPT-2")
        if memory_mask.shape != memory.shape[:2] or memory_mask.dtype != torch.bool:
            raise ValueError("memory_mask must be boolean [N, S]")
        if not memory_mask.any(-1).all():
            raise ValueError("every regional sample needs one memory token")

    def _forward_text(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        targets: Sequence[str],
        *,
        max_length: int,
    ) -> dict:
        self._validate(memory, memory_mask)
        if len(targets) != memory.shape[0]:
            raise ValueError("one target text is required per decoder sample")
        ids, attention, labels, truncated = pack_phrases(
            self.tokenizer, targets, max_length, memory.device
        )
        output = self.lm(
            input_ids=ids,
            attention_mask=attention.long(),
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask.long(),
            use_cache=False,
            return_dict=True,
        )
        loss_sums, token_counts = token_ce_sums(output.logits, labels)
        return {
            "loss": loss_sums.sum() / token_counts.sum().clamp_min(1),
            "token_count": token_counts.sum(),
            "loss_sums": loss_sums,
            "token_counts": token_counts,
            "truncated": truncated,
        }

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        phrases: Sequence[str],
    ) -> dict:
        """Teacher-force one regional phrase per memory row."""
        return self._forward_text(
            memory,
            memory_mask,
            phrases,
            max_length=self.max_length,
        )

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        max_new_tokens: int | None = None,
        num_beams: int | None = None,
        length_penalty: float | None = None,
    ) -> list[str]:
        self._validate(memory, memory_mask)
        tokens = self.max_length + 1 if max_new_tokens is None else int(max_new_tokens)
        if tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        beams = self.num_beams if num_beams is None else int(num_beams)
        penalty = self.length_penalty if length_penalty is None else float(length_penalty)
        prompt = torch.full(
            (memory.shape[0], 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=memory.device,
        )
        kwargs = {
            "input_ids": prompt,
            "attention_mask": torch.ones_like(prompt),
            "encoder_hidden_states": memory,
            "encoder_attention_mask": memory_mask.long(),
            "max_new_tokens": tokens,
            "num_beams": beams,
            "do_sample": False,
            "early_stopping": beams > 1,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if beams > 1:
            kwargs["length_penalty"] = penalty
        output = self.lm.generate(**kwargs)
        return self.tokenizer.batch_decode(output[:, 1:], skip_special_tokens=True)

__all__ = [
    "RegionalDecoder",
    "load_pubmed_gpt",
    "pack_phrases",
    "token_ce_sums",
    "token_level_ce",
]
