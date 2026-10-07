# Evaluation

Every version of Lyra is evaluated with the same command, and the result is stored as JSON in `evaluation/results/`, so versions can be compared objectively.

## Running

```bash
python -m scripts.evaluate --name lyra-0.1
python -m scripts.evaluate --name lyra-0.2 --checkpoint checkpoints/<run>/best.pt --compare evaluation/results/lyra-0.1.json
```

The second command prints each metric next to the previous version with a status of improved, worse, same or info.

## Metrics

| Group | Metric | Meaning | Better |
|-------|--------|---------|--------|
| Language model | Loss, perplexity | Next-token prediction on the whole validation and test splits (non-overlapping windows, no sampling) | Lower |
| Language model | Bits per byte | Total log-loss divided by the bytes of text predicted; independent of vocabulary size | Lower |
| Generated text | Distinct-1 and distinct-2 | Share of unique words and word pairs; low values mean repetition | Higher |
| Generated text | Invented word rate | Share of generated words that never appear in the training corpus | Lower |
| Generated text | Invalid byte rate | Share of texts containing invalid UTF-8 sequences | Lower |
| Generated text | Copy rate (6-grams) | Share of generated 6-word sequences found verbatim in the corpus; indicates memorization | Informational |
| Chat | Turn marker leaks | Replies that contain a fake turn label | Lower |
| Chat | Empty reply rate | Replies with no text | Lower |
| Chat | Prompt within budget | Whether the prompt always fit the model context | Yes |

Generation checks use 10 fixed prompts and a 6-message chat script (`configs/eval_prompts.json`) with fixed random seeds. The prompts must not change between versions.

## Baseline: Lyra 0.1

Checkpoint at step 1,500 (best validation loss), 995,968 parameters.

| Metric | Validation | Test |
|--------|-----------:|-----:|
| Loss | 4.3174 | 5.0192 |
| Perplexity | 75.0 | 151.3 |
| Bits per byte | 2.147 | 2.367 |
| Tokens evaluated | 6,656 | 6,656 |

| Check | Result |
|-------|-------:|
| Distinct-1 | 0.48 |
| Distinct-2 | 0.88 |
| Invented word rate | 10.9% |
| Copy rate (6-grams) | 0.0% |
| Invalid byte rate | 0% |
| Chat turn marker leaks | 0 |
| Chat empty reply rate | 0% |
| Chat prompt within budget | Yes |

## Limitations

- The test split is the end of the source file and contains the book's chapter index, so the test numbers are worse and less representative than the validation numbers. Track validation.
- The generated-text metrics come from about 600 words, so small differences between versions are within noise. Loss metrics are exact.
- Generated-text results are reproducible on the same machine and PyTorch version; they may differ slightly on other hardware.
- When the training corpus changes, the validation split changes too, so loss values are not directly comparable. The *Dom Casmurro* validation and test splits should be kept as a fixed reference set when new data is added.
- The suite does not yet measure factual accuracy, reasoning, tool use, web research, security behavior or prompt-injection resistance. Those need the corresponding features and are planned with later phases.
- No metric here measures whether replies are useful. The base model has not been trained for dialogue, and its replies continue the training text.