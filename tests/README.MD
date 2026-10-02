# Lyra

A compact, modular language model built from scratch: custom tokenizer, Transformer core, training pipeline and tool-integration layer, with no dependency on any pretrained model.

## Status

Early development (Lyra 0.1). The project is built in incremental phases, and each phase is tested before the next one starts.

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Byte-level BPE tokenizer | Complete |
| 2 | Neural core and text generation | Complete |
| 3 | Dataset and training pipeline | Planned |
| 4 | Conversation and context management | Planned |
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

## Project structure

```
lyra/
├── core/
│   ├── tokenizer/    BPE tokenizer
│   ├── model/        Transformer components and model
│   └── generation/   Sampling and text generation
├── tests/            Automated tests
├── scripts/          Utility scripts (data cleaning, tokenizer training)
├── configs/
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

To be defined.