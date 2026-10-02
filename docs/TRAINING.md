# Training

## Data

| Item | Value |
|------|-------|
| Corpus | *Dom Casmurro* by Machado de Assis (public domain, Project Gutenberg ebook 55752) |
| Cleaning | Gutenberg header and footer removed, Unicode NFC normalization, paragraph deduplication (paragraphs of 80+ characters) |
| Size | 380,928 characters, 133,610 tokens (2.85 characters per token) |
| Splits | Contiguous: 120,250 train, 6,680 validation, 6,680 test |

Deduplication removed nothing from this corpus. The spelling is early 20th century Brazilian Portuguese.

Datasets are never stored in the repository. The `data/` directory is ignored by Git and rebuilt with the scripts below.

## Configuration

All settings live in `configs/train_v0.json` and are loaded into `TrainConfig`. Unknown keys are rejected.

| Setting | Value |
|---------|-------|
| Seed | 1337 |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.01 (matrices only) |
| Learning rate | 1e-3 with 100 warmup steps, cosine decay to 1e-4 |
| Batch | 16 sequences x 2 gradient accumulation steps x 128 tokens (4,096 tokens per step) |
| Gradient clipping | 1.0 |
| Evaluation | Every 100 steps on 20 fixed validation batches |
| Early stopping | Patience of 5 evaluations |
| Mixed precision | bfloat16, enabled only when a CUDA device is available (not used in the baseline run) |

## Checkpoints

Written to `checkpoints/<run>/`:

- `best.pt`: weights with the lowest validation loss,
- `last.pt`: most recent state, used to resume,
- `metrics.jsonl`: one record per evaluation.

Each checkpoint stores model weights, optimizer state, step, best validation loss, model and training configuration, seed, dataset metadata, data-sampler random state and the PyTorch version. Files are written atomically (temporary file, then replace). Pausing and resuming is covered by a test that checks it produces the same weights as an uninterrupted run.

## Reproducing the baseline

```bash
python scripts/clean_text.py          # from data/raw to data/processed
python -m scripts.train_tokenizer     # trains the BPE tokenizer
python -m scripts.prepare_data        # tokenizes and stores token ids
python -m scripts.train               # trains the model
python -m scripts.train --resume      # resumes from checkpoints/.../last.pt
python -m scripts.sample --prompt "Uma noite destas"
```

## Baseline results (Lyra 0.1)

Run on a CPU-only laptop with 4 GB of RAM.

| Metric | Value |
|--------|-------|
| Steps | 2,000 (about 40 minutes) |
| Best validation loss | 4.3016 at step 1,500 (perplexity about 74) |
| Final training loss | About 3.4 |
| Test loss | 5.0217 (perplexity about 152) |
| Random-guess reference | Loss 7.31 (perplexity 1,500) |

Validation loss stopped improving after step 1,500 while training loss kept falling, which indicates the model starts memorizing the single book. The best checkpoint is kept.

## Limitations

- One book is a very small corpus; the model overfits after roughly 100 passes over the data.
- The test split is the final part of the file and includes the book's chapter index, so the test loss is not a clean quality estimate.
- Training is single-device. Nothing in the design prevents moving to PyTorch distributed training later, but it is not implemented.
- There is no benchmark suite yet beyond loss and perplexity; see the roadmap.