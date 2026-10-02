# Lyra

A compact, modular language model built from scratch: custom tokenizer, Transformer core, training pipeline and tool-integration layer, with no dependency on any pretrained model.

## Status

Early development (Lyra 0.1). The project is built in incremental phases, and each phase is tested before the next one starts.

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Byte-level BPE tokenizer | Complete |
| 2 | Neural core and text generation | Complete |
| 3 | Dataset and training pipeline | Complete |
| 4 | Conversation and context management | Complete (base model not yet      trained for dialogue) |
| 5 | Tool system | Planned |
| 6 | Web research | Planned |
| 7 | Memory | Planned |
| 8 | Integration API | Planned |
| 9 | Chat interface | Planned |
| 10 | Avatar | Planned |

## Design principles

- **Independent core.** Lyra's language model is trained from scratch. No external or pretrained language model is used as its brain.
- **Modular layers.** Tokenizer, model, training, inference, tools and integrations are separate components with clear responsibilities.
- **Small and measurable.** Start with a small model that can be trained on commodity hardware, and evaluate every version objectively.
- **Security by design.** Least-privilege tools, untrusted external content, and no direct database access by the model.
- **Zero or minimal cost.** Open-source and local-first tooling.

## Current architecture

| Component | Description |
|-----------|-------------|
| Tokenizer | Byte-level BPE, trainable on any corpus, with persistence to JSON |
| Model | Causal Transformer, pre-norm (RMSNorm), learned positional embeddings, tied input/output embeddings |
| Default size | About 1.0M parameters (4 layers, 4 heads, d_model 128, context 128) |
| Generation | Autoregressive sampling with temperature, top-k, top-p, max tokens and stop tokens |

## Results (Lyra 0.1 baseline)

Trained on a single public-domain novel (about 120k training tokens) on a laptop CPU, with no GPU.

| Metric | Value |
|--------|-------|
| Parameters | 995,968 |
| Vocabulary | 1,500 tokens (byte-level BPE) |
| Training time | About 40 minutes (CPU) |
| Best validation loss | 4.30 (perplexity about 74; random guessing is 1,500) |
| Test loss | 5.02 (perplexity about 152) |

The test split is the final 5% of the book, which includes a chapter index, so it is not a clean estimate of quality. The model reproduces the style, spelling and dialogue structure of the corpus but does not stay coherent beyond a short phrase. It is a baseline to be improved with more data and a larger model.

## Project structure

```
lyra/
├── core/
│   ├── tokenizer/    BPE tokenizer
│   ├── model/        Transformer components and model
│   ├── generation/   Sampling controls (temperature, top-k, top-p)
│   ├── inference/    Checkpoint loading and text generation
│   └── context/      Message structures and context budget manager
├── personality/      Configurable persona (no weight changes)
├── conversation/     Chat session tying persona, context and model together
├── training/         Data pipeline, trainer and checkpoints
├── tests/            Automated tests
├── scripts/          Data preparation, training, sampling and chat scripts
├── configs/          Training and persona configuration
├── docs/
└── data/             Local only, not versioned
```

## Getting started

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
python -m pytest -q
```

For a CPU-only PyTorch build (smaller download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Roadmap

- **0.1** Tokenizer, neural core, training pipeline, basic conversation
- **0.2** Tool system and web research
- **0.3** Memory and integration API
- **1.0** Stable core with evaluation benchmarks, SDK and interface

## License

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Model](docs/MODEL.md)
- [Training](docs/TRAINING.md)
- [Roadmap](docs/ROADMAP.md)
- [Changelog](CHANGELOG.md)