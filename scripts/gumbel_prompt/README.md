# Gumbel Prompt Scripts

This folder contains the end-to-end workflow for SAM3 prompt tuning with
Gumbel-Softmax, plus evaluation and inference utilities.

## Stages

1. Train/Tune: `optimize_sam3_prompt_gumbel.py`
   - Legacy-compatible alternative: `optimize_sam3_prompt_gumbel_v1_compat.py`
2. Eval: `eval.py`
3. Inference (single image): `inference.py`
4. Inference (batch, multi-prompt): `inference_multi_prompt_batch.py`

Artifacts from tuning are written under:

- `<out-dir>/<timestamp>/prompt_artifacts/` (best-soft, backward-compatible path)
- `<out-dir>/<timestamp>/prompt_artifacts/best_soft/`
- `<out-dir>/<timestamp>/prompt_artifacts/best_hard/`

Each training run creates a new timestamp-named subfolder (for example,
`20260314_154210`) to avoid overwriting previous experiments.

---

## 1) Train/Tune Prompt (`optimize_sam3_prompt_gumbel.py`)

Optimizes prompt token logits against mask IoU while SAM3 weights remain frozen.

### Example commands

Actual tuned run (seeded + gentle seed KL + hybrid refine):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/optimize_sam3_prompt_gumbel.py \
  --data-root ./datasets/train \
  --image-name 1202_jpeg_jpg.rf.0d023f64c50b4a15557e54572d3c5d0c.jpg \
  --prompt-len 4 \
  --steps 250 \
  --lr 0.05 \
  --forward-mode hard_st_only \
  --temperature-mode learnable_per_position \
  --tau-min 0.01 \
  --tau-init 1.0 \
  --tau-max 3.0 \
  --entropy-lambda 0.002 \
  --entropy-schedule ramp \
  --entropy-start-step 80 \
  --entropy-ramp-steps 80 \
  --seed-phrases "airplane,small plane" \
  --init-logit-bias 2.0 \
  --seed-kl-lambda 0.001 \
  --seed-kl-label-smoothing 0.02 \
  --seed-kl-decay-fraction 0.4 \
  --discrete-refine-every 20 \
  --discrete-top-m 3 \
  --discrete-max-candidates 32 \
  --discrete-min-delta 0.001 \
  --discrete-accept-mode off \
  --vocab-top-k 2000 \
  --save-every 25 \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt
```

Fixed schedule temperature (default):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/optimize_sam3_prompt_gumbel.py \
  --data-root ./datasets/train \
  --image-name 1202_jpeg_jpg.rf.0d023f64c50b4a15557e54572d3c5d0c.jpg \
  --prompt-len 6 \
  --steps 200 \
  --lr 0.05 \
  --tau-start 2.0 \
  --tau-end 0.2 \
  --vocab-top-k 2000 \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt
```

Learnable per-position temperature (recommended):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/optimize_sam3_prompt_gumbel.py \
  --data-root ./datasets/train \
  --image-name 1202_jpeg_jpg.rf.0d023f64c50b4a15557e54572d3c5d0c.jpg \
  --prompt-len 6 \
  --steps 200 \
  --lr 0.05 \
  --temperature-mode learnable_per_position \
  --tau-min 0.05 \
  --tau-init 1.0 \
  --tau-max 3.0 \
  --vocab-top-k 2000 \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt
```

Late-phase entropy regularization (optional):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/optimize_sam3_prompt_gumbel.py \
  --data-root ./datasets/train \
  --image-name 1202_jpeg_jpg.rf.0d023f64c50b4a15557e54572d3c5d0c.jpg \
  --prompt-len 6 \
  --steps 200 \
  --lr 0.05 \
  --temperature-mode learnable_per_position \
  --tau-min 0.05 \
  --tau-init 1.0 \
  --entropy-lambda 0.01 \
  --entropy-schedule late \
  --entropy-start-step 120 \
  --vocab-top-k 2000 \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt
```

Seeded initialization from phrase(s) (optional):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/optimize_sam3_prompt_gumbel.py \
  --data-root ./datasets/train \
  --image-name 1202_jpeg_jpg.rf.0d023f64c50b4a15557e54572d3c5d0c.jpg \
  --prompt-len 6 \
  --steps 200 \
  --lr 0.05 \
  --temperature-mode learnable_per_position \
  --tau-min 0.05 \
  --tau-init 1.0 \
  --seed-phrases "airplane,small plane" \
  --init-logit-bias 2.0 \
  --seed-kl-lambda 0.001 \
  --seed-kl-label-smoothing 0.02 \
  --seed-kl-decay-fraction 0.4 \
  --vocab-top-k 2000 \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt
```

Hybrid differentiable + discrete refinement (optional, disabled by default):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/optimize_sam3_prompt_gumbel.py \
  --data-root ./datasets/train \
  --image-name 1202_jpeg_jpg.rf.0d023f64c50b4a15557e54572d3c5d0c.jpg \
  --prompt-len 6 \
  --steps 200 \
  --lr 0.05 \
  --seed-phrases "airplane,small plane" \
  --init-logit-bias 2.0 \
  --discrete-refine-every 20 \
  --discrete-top-m 3 \
  --discrete-max-candidates 32 \
  --discrete-min-delta 0.001 \
  --discrete-accept-mode off \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt
```

Dry-run config validation:

```bash
PYTHONPATH=. python scripts/gumbel_prompt/optimize_sam3_prompt_gumbel.py \
  --data-root ./datasets/train \
  --image-name 1202_jpeg_jpg.rf.0d023f64c50b4a15557e54572d3c5d0c.jpg \
  --checkpoint-path ~/staged_models/sam3 \
  --dry-run-config
```

### Parameters

- `--data-root`: dataset root with images
- `--coco-json`: COCO annotation path (default: `<data-root>/_annotations.coco.json`)
- `--image-name`: image file name(s) from COCO JSON; can repeat or be comma-separated
- `--ann-id`: optional annotation id filter
- `--category-id`: optional category id filter
- `--prompt-len`: number of optimized prompt tokens
- `--steps`: optimization steps
- `--lr`: Adam learning rate
- `--forward-mode`: `current` | `hard_st_only`
- `--gumbel-mode`: `soft` | `straight_through` (forward hard, backward soft)
- `--temperature-mode`: `fixed` | `learnable_global` | `learnable_per_position`
- `--tau-start`: start temperature for fixed mode schedule
- `--tau-end`: end temperature for fixed mode schedule
- `--tau-min`: minimum learnable temperature (`tau = tau_min + softplus(u)`)
- `--tau-max`: optional upper clamp for learnable temperature
- `--tau-init`: initial learnable temperature (defaults to `tau-start`)
- `--vocab-top-k`: restrict each position to top-K vocab logits (`0` disables)
- `--entropy-lambda`: entropy regularization weight (`0` disables)
- `--entropy-schedule`: `constant` | `late` | `ramp`
- `--entropy-start-step`: first step where late/ramp entropy applies
- `--entropy-ramp-steps`: warmup length for ramp schedule (`ramp` mode)
- `--allow-special-tokens`: allow special tokens in prompt slots (default blocks them)
- `--seed-phrases`: optional seed phrase or comma-separated phrases for logit init
- `--init-logit-bias`: additive bias for seeded token ids at initialization
- `--seed-kl-lambda`: optional seed-distribution regularization weight (`0` disables)
- `--seed-kl-label-smoothing`: smoothing applied in seed KL regularization
- `--seed-kl-decay-fraction`: fraction of total steps where seed KL decays to 0
- `--discrete-refine-every`: run bounded discrete hard-prompt refinement every K steps (`0` disables)
- `--discrete-top-m`: per-slot top-M ids used to build discrete candidates
- `--discrete-max-candidates`: max discrete candidates per refinement step
- `--discrete-min-delta`: required hard-IoU improvement for candidate acceptance
- `--discrete-accept-mode`: `off` | `bias_logits` (default `off`)
- `--discrete-accept-bias`: logit bias used when `discrete-accept-mode=bias_logits`
- `--eval-every-n-steps`: frequency for hard/soft eval logging
- `--save-every`: save periodic step outputs (`0` disables)
- `--save-best-only`: only save final best outputs
- `--amp-bf16`: run model forward with bf16 autocast
- `--checkpoint-path`: checkpoint `.pt` file or folder containing it
- `--device`: target device (default `cuda`)
- `--seed`: RNG seed
- `--out-dir`: output directory
- `--log-every`: log interval
- `--dry-run-config`: validate config and exit

### V1 Compatibility Trainer

Use `optimize_sam3_prompt_gumbel_v1_compat.py` when you want the original
fixed-temperature, soft-only optimization flow from the first implementation.
It keeps timestamped run directories but does not include newer training modes.

---

## 2) Evaluate Prompt (`eval.py`)

Compares text prompt vs saved soft prompt artifacts over COCO masks.

### Example commands

Side-by-side eval (text vs tuned soft prompt):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/eval.py \
  --data-root ./datasets/test \
  --prompt "airplane" \
  --soft-prompt-dir ./outputs/gumbel_prompt/<YYYYMMDD_HHMMSS>/prompt_artifacts/best_soft \
  --category-id 1 \
  --batch-size 4 \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt_eval \
  --save-outputs
```

Text-only eval (for baseline or V1-style text prompts):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/eval.py \
  --eval-mode text_only \
  --data-root ./datasets/test \
  --prompt "airplane" \
  --category-id 1 \
  --batch-size 4 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt_eval_text_only \
  --save-outputs
```

Compare hard-selected artifacts:

```bash
PYTHONPATH=. python scripts/gumbel_prompt/eval.py \
  --data-root ./datasets/test \
  --prompt "airplane" \
  --soft-prompt-dir ./outputs/gumbel_prompt/<YYYYMMDD_HHMMSS>/prompt_artifacts/best_hard \
  --category-id 1 \
  --batch-size 4 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/gumbel_prompt_eval_best_hard \
  --save-outputs
```

Dry-run config validation:

```bash
PYTHONPATH=. python scripts/gumbel_prompt/eval.py \
  --data-root ./datasets/test \
  --prompt "drone" \
  --soft-prompt-dir ./outputs/gumbel_prompt/prompt_artifacts \
  --checkpoint-path ~/staged_models/sam3 \
  --dry-run-config
```

### Parameters

- `--eval-mode`: `side_by_side` | `text_only`
- `--data-root`: dataset root with images
- `--coco-json`: COCO annotation path (default: `<data-root>/_annotations.coco.json`)
- `--prompt`: text prompt baseline
- `--soft-prompt-dir`: directory with soft prompt artifacts (required for `side_by_side`)
- `--image-name`: optional image file names; can repeat or comma-separated
- `--ann-id`: optional annotation id filter
- `--category-id`: optional category id filter
- `--batch-size`: eval batch size
- `--max-images`: cap number of evaluated images (`0` means all)
- `--device`: target device (default `cuda`)
- `--checkpoint-path`: checkpoint `.pt` file or folder containing it
- `--amp-bf16`: run model forward with bf16 autocast
- `--out-dir`: output directory
- `--save-outputs`: save per-image overlays and masks
- `--log-every`: progress logging period (in batches)
- `--dry-run-config`: validate config and exit

---

## 3) Inference, Single Image (`inference.py`)

Runs one image in text mode, `soft_saved` mode, or exact hard-token replay mode
(`hard_saved`).

### Example commands

Soft-saved prompt inference:

```bash
PYTHONPATH=. python scripts/gumbel_prompt/inference.py \
  --image-path ./datasets/test/a-55-_jpg.rf.06ae569493c67431acdcd4dfedf019af.jpg \
  --prompt-mode soft_saved \
  --soft-prompt-dir ./outputs/gumbel_prompt/<YYYYMMDD_HHMMSS>/prompt_artifacts/best_soft \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/inference
```

Text prompt inference:

```bash
PYTHONPATH=. python scripts/gumbel_prompt/inference.py \
  --image-path ./datasets/test/a-55-_jpg.rf.06ae569493c67431acdcd4dfedf019af.jpg \
  --prompt-mode text \
  --prompt "airplane" \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/inference
```

Hard-saved token-id inference (exact saved hard prompt replay):

```bash
PYTHONPATH=. python scripts/gumbel_prompt/inference.py \
  --image-path ./datasets/test/a-55-_jpg.rf.06ae569493c67431acdcd4dfedf019af.jpg \
  --prompt-mode hard_saved \
  --soft-prompt-dir ./outputs/gumbel_prompt/<YYYYMMDD_HHMMSS>/prompt_artifacts/best_hard \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/inference
```

### Parameters

- `--image-path`: input image path
- `--prompt-mode`: `text` | `soft_saved` | `hard_saved`
- `--prompt`: required when `prompt-mode=text`
- `--soft-prompt-dir`: required when `prompt-mode=soft_saved|hard_saved`
- `--checkpoint-path`: checkpoint `.pt` file or folder containing it
- `--device`: target device (default `cuda`)
- `--amp-bf16`: run model forward with bf16 autocast
- `--out-dir`: output directory

---

## 4) Inference, Batch Multi-Prompt (`inference_multi_prompt_batch.py`)

Runs multiple category prompts across a folder of images.

### Example command

```bash
PYTHONPATH=. python scripts/gumbel_prompt/inference_multi_prompt_batch.py \
  --data-root ./datasets/test \
  --prompt-item plane=airplane \
  --prompt-item jet=fighter jet \
  --prompt-item bird=bird \
  --batch-size 4 \
  --amp-bf16 \
  --checkpoint-path ~/staged_models/sam3 \
  --out-dir ./outputs/inference/multi_prompt
```

### Parameters

- `--data-root`: directory with images
- `--image-name`: optional subset of image names; can repeat or comma-separated
- `--prompt-item`: category prompt mapping, format `category=prompt` or `category:prompt`
- `--batch-size`: inference batch size
- `--checkpoint-path`: checkpoint `.pt` file or folder containing it
- `--device`: target device (default `cuda`)
- `--amp-bf16`: run model forward with bf16 autocast
- `--out-dir`: output directory

---

## Notes

- `prompt_artifacts/` can be used directly for `soft_saved` eval/inference.
- Tuning stores both best-soft and best-hard selections.
- During tuning, periodic/final visual outputs are saved for both `soft_saved`
  and `hard` prompt modes under per-step directories.
- Final best visual outputs are saved under each image at `best/soft_saved` and `best/hard`.
- If discrete refinement is enabled, a run-level `refinement_summary.json` is saved
  in `<out-dir>/<timestamp>/`.
- For best compatibility, keep the same model/checkpoint family between tuning,
  eval, and inference.
