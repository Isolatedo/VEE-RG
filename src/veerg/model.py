from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import RegionalDecoder, load_pubmed_gpt
from .event import EventAdapter
from .longitudinal.frozen import FrozenLongitudinalEncoder
from .visual import (
    FrozenAnatomicalDetector,
    FrozenRegionSelector,
    RegionalVisionEncoder,
    RegionalVisualAdapter,
)


CHECKPOINT_ARCH = "vee-rg-report-v1"
NUM_REGIONS = 29
WEIGHT_DECAY = 0.01


class VEEReportGenerator(nn.Module):
    """Generate a report by decoding one selected phrase per anatomy region."""

    def __init__(
        self,
        vision: RegionalVisionEncoder,
        decoder: RegionalDecoder,
        longitudinal: FrozenLongitudinalEncoder,
        event_adapter: EventAdapter,
    ):
        super().__init__()
        self.vision = vision
        self.decoder = decoder
        self.longitudinal = longitudinal
        self.event_adapter = event_adapter
        self.stage = "report_generation"

    @property
    def visual_adapter(self) -> RegionalVisualAdapter:
        return self.vision.adapter

    @property
    def lm(self):
        return self.decoder.lm

    def configure_training(self) -> None:
        """Freeze the detector, selector, and longitudinal encoder for Stage 2."""
        self.requires_grad_(False)
        self.visual_adapter.requires_grad_(True)
        self.event_adapter.requires_grad_(True)
        for name, parameter in self.lm.named_parameters():
            if ".crossattention." in name or ".ln_cross_attn." in name:
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.detector.eval()
        self.vision.selector.eval()
        self.longitudinal.eval()
        self.lm.eval()
        if mode:
            self.visual_adapter.train()
            for block in self.lm.transformer.h:
                block.crossattention.train()
                block.ln_cross_attn.train()
            self.event_adapter.train()
        return self

    def encode_memory(self, batch: dict) -> dict[str, torch.Tensor]:
        encoded = self.vision(batch["current_image"])
        spatial = encoded.pop("spatial")
        has_prior = batch["has_real_prior"]
        event_state = self.longitudinal(
            spatial,
            batch["prior_image"],
            has_prior,
        )
        valid = event_state["valid"] & encoded["roi_mask"]
        _, event_delta, event_mask = self.event_adapter(
            encoded["roi_tokens"],
            event_state["exchange_invariant"],
            event_state["order_sensitive"],
            valid,
            has_prior,
        )

        # Each selected region receives ROI, EVENT, global C5, and identity slots.
        global_tokens = encoded["global_token"].unsqueeze(1).expand(
            -1, NUM_REGIONS, -1
        )
        local_memory = torch.stack(
            (
                encoded["roi_tokens"],
                event_delta,
                global_tokens,
                encoded["region_tokens"],
            ),
            dim=2,
        )
        local_mask = torch.stack(
            (
                encoded["roi_mask"],
                event_mask,
                torch.ones_like(encoded["roi_mask"]),
                torch.ones_like(encoded["roi_mask"]),
            ),
            dim=2,
        )
        encoded.update(
            local_memory=local_memory,
            local_memory_mask=local_mask,
            event_delta=event_delta,
            event_mask=event_mask,
            event_state=event_state,
            spatial=spatial,
        )
        return encoded

    @staticmethod
    def _flatten(encoded: dict, mask: torch.Tensor):
        if mask.shape != encoded["local_memory"].shape[:2] or mask.dtype != torch.bool:
            raise ValueError("regional mask must be boolean [B, 29]")
        indices = mask.nonzero(as_tuple=False)
        if indices.numel() == 0:
            return None
        image_indices, region_indices = indices.unbind(-1)
        memory = encoded["local_memory"][image_indices, region_indices]
        memory_mask = encoded["local_memory_mask"][image_indices, region_indices]
        return indices, memory, memory_mask

    @staticmethod
    def _zero_decoder_output(encoded: dict) -> dict:
        zero = encoded["roi_tokens"].sum() * 0.0
        if not zero.requires_grad:
            zero = torch.zeros((), device=zero.device, requires_grad=True)
        return {
            "loss": zero,
            "token_count": torch.zeros((), dtype=torch.long, device=zero.device),
            "loss_sums": torch.empty(0, dtype=zero.dtype, device=zero.device),
            "token_counts": torch.empty(0, dtype=torch.long, device=zero.device),
            "truncated": 0,
        }

    def forward(self, batch: dict) -> dict:
        encoded = self.encode_memory(batch)
        target_mask = batch["region_has_sentence"] & encoded["roi_mask"]
        flattened = self._flatten(encoded, target_mask)
        if flattened is None:
            regional_output = self._zero_decoder_output(encoded)
            image_indices = torch.empty(
                0, dtype=torch.long, device=encoded["roi_tokens"].device
            )
            loss_sums = regional_output["loss_sums"]
            token_counts = regional_output["token_counts"]
        else:
            indices, memory, memory_mask = flattened
            image_indices = indices[:, 0]
            phrases = [
                batch["region_phrases"][int(image)][int(region)]
                for image, region in indices.tolist()
            ]
            regional_output = self.decoder(memory, memory_mask, phrases)
            loss_sums = regional_output.pop("loss_sums")
            token_counts = regional_output.pop("token_counts")

        output = {
            "loss": regional_output["loss"],
            "regional_loss": regional_output["loss"],
            "regional_loss_sum": loss_sums.sum(),
            "regional_token_count": regional_output["token_count"],
            "token_count": regional_output["token_count"],
            "truncated": regional_output.get("truncated", 0),
        }
        regional_prior = batch["has_real_prior"].index_select(0, image_indices)
        for name, group in (
            ("real_prior", regional_prior),
            ("no_prior", ~regional_prior),
        ):
            output[f"{name}_loss_sum"] = loss_sums[group].sum().detach()
            output[f"{name}_token_count"] = token_counts[group].sum().detach()
        output.update(encoded)
        output["supervised_regions"] = int(target_mask.sum().item())
        output["selected_regions"] = int(encoded["selected_mask"].sum().item())
        return output

    @torch.no_grad()
    def _generate_from_encoded(self, encoded: dict):
        selected = encoded["selected_mask"]
        flattened = self._flatten(encoded, selected)
        sentences = [[""] * NUM_REGIONS for _ in range(selected.shape[0])]
        if flattened is not None:
            indices, memory, memory_mask = flattened
            generated = self.decoder.generate(memory, memory_mask)
            for (image, region), sentence in zip(indices.tolist(), generated):
                sentences[image][region] = sentence
        return sentences, selected.tolist()

    @torch.no_grad()
    def generate_regions(self, batch: dict):
        encoded = self.encode_memory(batch)
        return self._generate_from_encoded(encoded)

    @torch.no_grad()
    def generate(self, batch: dict) -> list[str]:
        from .assembly import assemble_batch

        sentences, selected = self.generate_regions(batch)
        reports, _ = assemble_batch(sentences, selected)
        return reports


def build_model(cfg) -> VEEReportGenerator:
    tokenizer, language_model = load_pubmed_gpt(
        cfg.model.language_model_path,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )
    hidden_dim = int(language_model.config.n_embd)
    if hidden_dim != cfg.model.hidden_dim:
        raise ValueError(
            f"hidden_dim={cfg.model.hidden_dim} does not match GPT-2 n_embd={hidden_dim}"
        )
    detector = FrozenAnatomicalDetector(cfg.model.detector_checkpoint)
    vision = RegionalVisionEncoder(
        detector,
        FrozenRegionSelector(cfg.model.selector_checkpoint),
        RegionalVisualAdapter(hidden_dim=hidden_dim, dropout=cfg.model.dropout),
        selector_threshold=cfg.model.selector_threshold,
        selector_top_k=cfg.model.selector_top_k,
    )
    longitudinal = FrozenLongitudinalEncoder(
        detector,
        cfg.model.longitudinal_checkpoint,
        image_size=cfg.data.image_size,
    )
    model = VEEReportGenerator(
        vision,
        RegionalDecoder(
            tokenizer,
            language_model,
            max_length=cfg.model.max_length,
            num_beams=cfg.model.num_beams,
            length_penalty=cfg.model.length_penalty,
        ),
        longitudinal,
        EventAdapter(
            hidden_dim=hidden_dim,
            order_scale=cfg.model.order_sensitive_scale,
        ),
    )
    model.configure_training()
    return model


def build_optimizer(model: VEEReportGenerator, cfg):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=cfg.train.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )


__all__ = ["CHECKPOINT_ARCH", "VEEReportGenerator", "build_model", "build_optimizer"]
