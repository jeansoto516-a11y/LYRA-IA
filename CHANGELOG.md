# Changelog

All notable changes to this project are documented here.

## Unreleased (Lyra 0.1)

### Added
- Byte-level BPE tokenizer with training, encode/decode and JSON persistence.
- Causal Transformer model (RMSNorm, learned positions, tied embeddings) and sampling with temperature, top-k, top-p, max tokens and stop tokens.
- Data pipeline: normalization, paragraph deduplication, tokenization, train/validation/test split and batch sampling.
- Training system: external JSON configuration, AdamW, warmup and cosine schedule, gradient accumulation, gradient clipping, validation, early stopping, atomic checkpoints and resumable training.
- Inference engine that loads a checkpoint and a tokenizer and generates text, independent of the training code.
- Scripts for cleaning text, training the tokenizer, preparing data, training and sampling.
- Automated test suite (55 tests).
- Baseline result: validation loss 4.30 (perplexity about 74) on a single public-domain novel.