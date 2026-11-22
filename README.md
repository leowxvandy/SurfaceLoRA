# SurfaceLoRA

This repository provides the reference implementation of **SurfaceLoRA** described in the paper **“Vectors Are Not Neutral: Sensitive-Information Inference from Exported LLM Representations in Summarization.”**

## Overview

SurfaceLoRA studies sensitive-label recoverability from a designated prompt representation exported by an LLM summarization system. It combines supervised summarization fine-tuning with a prompt-only adversarial branch:

- the pretrained language-model backbone remains frozen;
- LoRA adapters and a training-time sensitive-label discriminator are trainable;
- a gradient reversal layer sends the negative, scaled discriminator gradient into the adapters;
- each training iteration pairs one utility batch with one adversarial batch drawn from a globally class-balanced audit set.

The main audit uses five EHR-recorded administrative race groups: White, Black, Hispanic, Asian, and Other. Unknown or ambiguous values are excluded. These labels are an administrative case study, not biological attributes.

SurfaceLoRA is artifact-specific. `lasttok` is the final non-padding prompt-token state immediately before generation. `meanpool` averages all non-padding prompt-token states. Reducing recoverability from one representation does not establish the same property for another representation or for the model as a whole.

## Method components

The package includes patient-group splitting, balanced audit-subset construction, chat-template prompting, LoRA summarization training, SurfaceLoRA training, attention-mask-aware representation extraction, independent linear and two-layer MLP post-hoc probes, ROUGE, LeakageGap, summary generation, and validation-only checkpoint selection.

The training-time discriminator and post-hoc probes are separate classes and never share learned weights. Post-hoc probes operate on frozen cached vectors and cannot update the language model.

## Repository structure

```text
configs/main.yaml          Main paper configuration
src/surfacelora/          Reference implementation
scripts/                  Data, training, extraction, evaluation, selection
tests/                    Focused synthetic unit tests
examples/                 Synthetic end-to-end smoke test
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[llm,test]"
```

The main configuration uses `meta-llama/Llama-3.1-8B-Instruct`. Obtain model access under the provider’s terms and configure an authorized local path or cache.

## Data preparation

Clinical datasets must be obtained through their official credentialed-access channels. No clinical text, identifiers, generated summaries, representations, or trained checkpoints are distributed here.

For MIMIC-IV-Ext-BHC v1.2.0, follow Appendix D of the paper. Join the BHC file to the MIMIC-IV-Note discharge table by `note_id` to obtain `subject_id` and `hadm_id`, then join the MIMIC-IV admissions table by `hadm_id` to obtain `race`. Keep one row per `note_id` and drop rows with missing identifiers, demographics, or text. Use a stable private key for each `subject_id` as `group_id`, and map the BHC file's `input` and `target` fields to `source` and `target`. The preparation command reads the resulting local JSONL file:

```json
{"group_id":"local-opaque-patient-key","source":"...","target":"...","race":"WHITE"}
```

`group_id` must be a private, stable local key used only to enforce patient-disjoint splits. Do not use or publish real identifiers. Accepted race values are mapped to the five configured labels; unknown values are excluded. Configure local paths in `configs/main.yaml`, then run:

```bash
export SURFACELORA_ALIGNMENT_SALT="choose-a-private-local-value"
python scripts/prepare_data.py --config configs/main.yaml
```

The salt produces one-way local alignment keys and must not be committed. The output contains protected text and is ignored by `.gitignore`; keep it in an approved local environment.

## Training

Baseline LoRA summarization:

```bash
python scripts/train_baseline.py --config configs/main.yaml
```

SurfaceLoRA targeting the configured exported representation:

```bash
python scripts/train_surfacelora.py \
  --config configs/main.yaml \
  --lambda-adv 0.02
```

The default prompt-only length is 1,024 tokens, the prompt-target SFT length is 1,536 tokens, and generation uses greedy decoding with at most 256 new tokens. GPU selection is controlled through the configuration or standard environment mechanisms; GPU IDs are not hard-coded.

## Representation evaluation

Extract aligned balanced training and validation representations for a checkpoint:

```bash
python scripts/extract_representations.py \
  --config configs/main.yaml \
  --representation lasttok \
  --adapter outputs/surfacelora/adapter_final \
  --checkpoint-id lambda-0.02-final \
  --output-dir outputs/representations/lambda-0.02-final
```

Use `--representation meanpool` to audit the pooled interface. The selected transformer layer is configured in `representation.layer` and defaults to the final layer.
Layer values name transformer blocks: `-1` is the final block (`L-1`), and the embedding output is never treated as block zero.

Create an aggregate validation record with the LR probe and the corresponding validation ROUGE-L:

Use `surfacelora.training.generate_summaries` and `surfacelora.metrics.rouge_scores` to compute aggregate validation ROUGE locally; generated clinical text must remain in the protected runtime environment.

```bash
python scripts/evaluate.py \
  --config configs/main.yaml \
  --representations outputs/representations/lambda-0.02-final \
  --split validation \
  --rouge-l 14.50 \
  --step 2000 \
  --output-dir outputs/validation/lambda-0.02-final
```

For the balanced five-class audit, chance accuracy is `0.20` and:

```text
LeakageGap = abs(probe_accuracy - 0.20)
```

## Checkpoint selection

Checkpoint selection accepts aggregate balanced-validation records only. It retains checkpoints with validation LR `LeakageGap <= 0.025` and selects the feasible checkpoint with the highest validation ROUGE-L. If none is feasible, it applies a deterministic validation Pareto fallback. Test metrics and row-level data are rejected.

```bash
python scripts/select_checkpoint.py \
  --metrics-dir outputs/validation
```

Each input JSON/CSV record must contain `checkpoint_identifier`, `split`, `representation`, `layer`, `lr_probe_accuracy`, `rougeL`, and optionally `step`.

Only after selection, extract the test representation for that checkpoint and perform final probing:

```bash
python scripts/extract_representations.py \
  --config configs/main.yaml \
  --representation lasttok \
  --adapter outputs/surfacelora/adapter_final \
  --checkpoint-id lambda-0.02-final \
  --output-dir outputs/representations/lambda-0.02-final \
  --splits test

python scripts/evaluate.py \
  --config configs/main.yaml \
  --representations outputs/representations/lambda-0.02-final \
  --split test \
  --output-dir outputs/final-evaluation
```

Final mode fits independent LR and MLP probes on frozen vectors; MLP early stopping uses validation only.

Refer to the paper for the reported empirical results and experimental analysis.

## Synthetic validation

```bash
pytest -q
python examples/run_synthetic_smoke_test.py
```

All examples and tests use synthetic text, labels, keys, and tensors.

## Citation

See `CITATION.cff`; update its bibliographic fields when the final publication metadata are available.

## License

Original code is released under the MIT License. Models, datasets, and dependencies remain subject to their own licenses and access terms.
