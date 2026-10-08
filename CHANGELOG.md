# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added
- Tool system: registry, typed schemas with strict validation, least-privilege permissions, executor with timeout, output limit and human confirmation for sensitive tools, structured audit log with secret redaction.
- Tool-call protocol (`<tool_call>` JSON) with escaping of tool results to prevent injection, and an agent loop with a step limit.
- Built-in tools: `calculate` (no `eval`), `get_current_time` and `read_text_file` (restricted to one authorized folder).
- `tool` message role in the conversation context.
- Demo script `scripts/tools_demo.py` (simulated model).
- Documentation: `docs/TOOLS.md` and `docs/SECURITY.md`.
- Automated tests: 191 in total.

## 0.1.0

### Added
- Byte-level BPE tokenizer with training, encode/decode and JSON persistence.
- Causal Transformer model (RMSNorm, learned positions, tied embeddings) and sampling with temperature, top-k, top-p, max tokens and stop tokens.
- Data pipeline: normalization, paragraph deduplication, tokenization, train/validation/test split and batch sampling.
- Training system: external JSON configuration, AdamW, warmup and cosine schedule, gradient accumulation, gradient clipping, validation, early stopping, atomic checkpoints and resumable training.
- Inference engine that loads a checkpoint and a tokenizer and generates text, independent of the training code.
- Scripts for cleaning text, training the tokenizer, preparing data, training and sampling.
- Conversation layer: message structures, a context manager that fits history into the model's token budget, a persona loaded from JSON, a chat session and a terminal chat script.
- Evaluation suite: exact language-model metrics (loss, perplexity, bits per byte), generated-text checks (repetition, invented words, memorization, invalid bytes), chat checks and a command to compare versions. Results are stored in `evaluation/results/`.
- Automated test suite (87 tests).
- Baseline result: validation loss 4.30 (perplexity about 74) on a single public-domain novel.