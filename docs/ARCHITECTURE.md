# Architecture

Lyra is organized as independent layers. The neural model contains no business logic, and external systems connect through adapters without changing the core.

## Layers and status

| Layer | Location | Responsibility | Status |
|-------|----------|----------------|--------|
| Tokenizer | `core/tokenizer/` | Text to token ids and back; trainable and persisted separately | Implemented |
| Neural model | `core/model/` | Causal Transformer, forward pass and loss | Implemented |
| Generation | `core/generation/` | Sampling controls (temperature, top-k, top-p, stop tokens) | Implemented |
| Inference engine | `core/inference/` | Load checkpoint and tokenizer, generate text | Implemented |
| Dataset pipeline | `training/dataset.py`, `scripts/` | Cleaning, deduplication, tokenization, splits, batching | Implemented |
| Training system | `training/` | Config, optimizer, schedule, evaluation, checkpoints, resume | Implemented |
| Evaluation | `tests/`, `docs/` | Loss and perplexity today; benchmark suite planned | Partial |
| Context manager | planned | Structured messages: system, history, tool results, memory | Planned |
| Personality | planned | Configurable behavior without changing weights | Planned |
| Tool system | planned | Registry, schemas, permissions, execution | Planned |
| Web research | planned | Search as a tool; external content treated as untrusted | Planned |
| Memory | planned | Session, persistent, contextual and operational memory | Planned |
| Security layer | planned | Least privilege, confirmations, audit | Planned |
| Integration API / SDK | planned | Versioned API (`/api/v1`), adapters for external systems | Planned |
| Interface and avatar | planned | Chat UI; avatar as a state-driven view with no intelligence | Planned |

## Dependency rules

```
interface / avatar
        |
   Integration API  ->  Adapters (system-specific code lives only here)
        |
  Context, Memory, Personality, Tool system, Security
        |
  Inference engine -> Generation -> Neural model
        |                                |
     Tokenizer                    (checkpoint files)

Training system -> Dataset pipeline, Neural model, Tokenizer
```

- The core (`core/`) never imports from `training/`. Inference loads a checkpoint file and does not depend on training scripts.
- The neural model does not know about tools, memory, databases or any product.
- Anything specific to a host system (for example a SaaS product) belongs in a separate adapter, never in the core.

## Design decisions

| Decision | Reason |
|----------|--------|
| Model written from scratch, no pretrained language model | Core requirement of the project: Lyra owns its model |
| Byte-level BPE tokenizer | Never produces unknown tokens, handles accents, URLs, numbers and code, small implementation |
| Small model first (about 1M parameters) | Must train on commodity CPU hardware and stay easy to measure |
| Training configuration in JSON, validated on load | Change experiments without touching code; unknown keys are rejected |
| Checkpoints store optimizer, sampler state, config and dataset metadata | Training can be paused and resumed exactly |
| Inference separated from training | Deployment should not need training code |
| Model learning separated from user memory | Conversations never change model weights automatically |

## Security principles (design intent for later phases)

- External content (web pages, tool output) is data, never instructions.
- Tools declare what they access and do; each call is checked against permissions.
- The model never receives unrestricted database access; writes need validated parameters and, when sensitive, human confirmation.
- Secrets are never logged.

None of these layers beyond the neural core, training and inference exist yet.