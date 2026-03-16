from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Tuple

import torch

from sam3.model.text_encoder_ve import VETextEncoder


def encode_soft_prompt(
    language_backbone: VETextEncoder,
    soft_tokens: torch.Tensor,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    encoder = language_backbone.encoder
    seq_len = token_ids.shape[1]

    inputs_embeds = encoder.token_embedding(token_ids)
    inputs_embeds = inputs_embeds.clone()
    token_embedding_table = encoder.token_embedding.weight
    soft_token_embeds = soft_tokens @ token_embedding_table
    inputs_embeds[:, 1 : 1 + soft_tokens.shape[0]] = soft_token_embeds.unsqueeze(0)

    attn_mask = encoder.attn_mask
    if attn_mask is not None:
        attn_mask = attn_mask[:seq_len, :seq_len]

    x = inputs_embeds + encoder.positional_embedding[:seq_len]
    x = encoder.transformer(x, attn_mask=attn_mask)
    x = encoder.ln_final(x)

    text_attention_mask = token_ids.eq(0)
    text_memory = x.transpose(0, 1)
    text_memory_resized = language_backbone.resizer(text_memory)

    return (
        text_attention_mask,
        text_memory_resized,
        {
            "inputs_embeds": inputs_embeds,
            "soft_token_embeds": soft_token_embeds,
        },
    )


def encode_hard_prompt(
    language_backbone: VETextEncoder,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    encoder = language_backbone.encoder
    seq_len = token_ids.shape[1]

    inputs_embeds = encoder.token_embedding(token_ids)
    attn_mask = encoder.attn_mask
    if attn_mask is not None:
        attn_mask = attn_mask[:seq_len, :seq_len]

    x = inputs_embeds + encoder.positional_embedding[:seq_len]
    x = encoder.transformer(x, attn_mask=attn_mask)
    x = encoder.ln_final(x)

    text_attention_mask = token_ids.eq(0)
    text_memory = x.transpose(0, 1)
    text_memory_resized = language_backbone.resizer(text_memory)
    return text_attention_mask, text_memory_resized, {"inputs_embeds": inputs_embeds}


def encode_soft_prompt_from_embeds(
    language_backbone: VETextEncoder,
    soft_embeds: torch.Tensor,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = language_backbone.encoder
    seq_len = token_ids.shape[1]

    inputs_embeds = encoder.token_embedding(token_ids)
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[:, 1 : 1 + soft_embeds.shape[0]] = soft_embeds.unsqueeze(0)

    attn_mask = encoder.attn_mask
    if attn_mask is not None:
        attn_mask = attn_mask[:seq_len, :seq_len]

    x = inputs_embeds + encoder.positional_embedding[:seq_len]
    x = encoder.transformer(x, attn_mask=attn_mask)
    x = encoder.ln_final(x)

    text_attention_mask = token_ids.eq(0)
    text_memory = x.transpose(0, 1)
    text_memory_resized = language_backbone.resizer(text_memory)
    return text_attention_mask, text_memory_resized, inputs_embeds


def load_soft_prompt_artifacts(
    soft_prompt_dir: str,
    language_backbone: VETextEncoder,
    device: torch.device,
) -> Tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    metadata_path = os.path.join(soft_prompt_dir, "prompt_metadata.json")
    soft_embeds_path = os.path.join(soft_prompt_dir, "best_soft_embeds.pt")
    soft_tokens_path = os.path.join(soft_prompt_dir, "best_soft_tokens.pt")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing prompt metadata: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if os.path.isfile(soft_embeds_path):
        soft_embeds = torch.load(soft_embeds_path, map_location="cpu")
    elif os.path.isfile(soft_tokens_path):
        soft_tokens = torch.load(soft_tokens_path, map_location="cpu")
        table = language_backbone.encoder.token_embedding.weight.detach().cpu()
        soft_embeds = soft_tokens @ table
    else:
        raise FileNotFoundError(
            "Missing soft prompt artifacts. Expected best_soft_embeds.pt or best_soft_tokens.pt"
        )

    if soft_embeds.ndim != 2:
        raise ValueError(
            f"Expected soft embeds shape [L, D], got {tuple(soft_embeds.shape)}"
        )

    embed_dim = int(language_backbone.encoder.width)
    if soft_embeds.shape[1] != embed_dim:
        raise ValueError(
            f"Soft embed dim mismatch: saved={soft_embeds.shape[1]} current={embed_dim}"
        )

    prompt_len = int(metadata["prompt_len"])
    if soft_embeds.shape[0] != prompt_len:
        raise ValueError(
            f"Soft prompt length mismatch: saved_tensors={soft_embeds.shape[0]} metadata={prompt_len}"
        )

    context_length = int(language_backbone.context_length)
    token_ids = torch.zeros((1, context_length), dtype=torch.long, device=device)
    token_ids[0, 0] = int(metadata["sot_token_id"])
    token_ids[0, 1 : 1 + prompt_len] = int(metadata.get("pad_token_id", 0))
    token_ids[0, 1 + prompt_len] = int(metadata["eot_token_id"])

    return soft_embeds.to(device=device, dtype=torch.float32), metadata, token_ids


def load_hard_prompt_artifacts(
    prompt_dir: str,
    language_backbone: VETextEncoder,
    device: torch.device,
) -> Tuple[torch.Tensor, dict[str, Any]]:
    metadata_path = os.path.join(prompt_dir, "prompt_metadata.json")
    hard_tokens_path = os.path.join(prompt_dir, "best_hard_token_ids.pt")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing prompt metadata: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if os.path.isfile(hard_tokens_path):
        hard_token_ids = torch.load(hard_tokens_path, map_location="cpu")
    elif "best_token_ids" in metadata:
        hard_token_ids = torch.tensor(metadata["best_token_ids"], dtype=torch.long)
    else:
        raise FileNotFoundError(
            "Missing hard prompt artifacts. Expected best_hard_token_ids.pt or best_token_ids in metadata"
        )

    if hard_token_ids.ndim != 1:
        raise ValueError(
            f"Expected hard token ids shape [L], got {tuple(hard_token_ids.shape)}"
        )

    prompt_len = int(metadata["prompt_len"])
    if hard_token_ids.shape[0] != prompt_len:
        raise ValueError(
            f"Hard prompt length mismatch: saved={hard_token_ids.shape[0]} metadata={prompt_len}"
        )

    context_length = int(language_backbone.context_length)
    token_ids = torch.zeros((1, context_length), dtype=torch.long, device=device)
    token_ids[0, 0] = int(metadata["sot_token_id"])
    token_ids[0, 1 : 1 + prompt_len] = hard_token_ids.to(device=device)
    token_ids[0, 1 + prompt_len] = int(metadata["eot_token_id"])

    return token_ids, metadata


def save_prompt_artifacts(
    artifacts_dir: str,
    best_logits: torch.Tensor,
    best_tau: torch.Tensor,
    best_soft_tokens: torch.Tensor,
    best_soft_embeds: torch.Tensor,
    best_hard_tokens: Optional[torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)

    logits_path = os.path.join(artifacts_dir, "best_prompt_logits.pt")
    tau_path = os.path.join(artifacts_dir, "best_tau.pt")
    soft_tokens_path = os.path.join(artifacts_dir, "best_soft_tokens.pt")
    soft_embeds_path = os.path.join(artifacts_dir, "best_soft_embeds.pt")
    hard_tokens_path = os.path.join(artifacts_dir, "best_hard_token_ids.pt")
    meta_path = os.path.join(artifacts_dir, "prompt_metadata.json")

    torch.save(best_logits, logits_path)
    torch.save(best_tau, tau_path)
    torch.save(best_soft_tokens, soft_tokens_path)
    torch.save(best_soft_embeds, soft_embeds_path)
    if best_hard_tokens is not None:
        torch.save(best_hard_tokens, hard_tokens_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Saved soft prompt artifact: %s", logits_path)
    logging.info("Saved soft prompt artifact: %s", tau_path)
    logging.info("Saved soft prompt artifact: %s", soft_tokens_path)
    logging.info("Saved soft prompt artifact: %s", soft_embeds_path)
    if best_hard_tokens is not None:
        logging.info("Saved soft prompt artifact: %s", hard_tokens_path)
    logging.info("Saved soft prompt artifact: %s", meta_path)
