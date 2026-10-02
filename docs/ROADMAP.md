# Roadmap

## Lyra 0.1: own core, trained baseline

- [x] Byte-level BPE tokenizer with tests
- [x] Causal Transformer, generation with sampling controls
- [x] Data pipeline and training system with checkpoints and resume
- [x] Inference engine
- [x] Baseline result: validation loss 4.30 (perplexity about 74) on one novel
- [x] Conversation basics: message and context structures, history, persona
- [ ] Minimal evaluation set so versions can be compared

## Lyra 0.2: more data, tools and web

- Larger Portuguese corpus with verified licenses; retrain the tokenizer on it
- Larger model configuration (tied to available hardware)
- Tool system: registry, schemas, permissions, execution and logging
- Web search as a tool with source tracking and untrusted-content handling

## Lyra 0.3: memory and integration

- Memory layers: session, persistent, contextual, operational
- Versioned API (`/api/v1`), authentication, sessions, adapters
- Evaluation benchmarks covering Portuguese, reasoning, tool use, safety and hallucination

## Lyra 1.0: stable core

- Stable model, tokenizer and API with documented benchmarks per version
- SDK for integrating external systems
- Chat interface and avatar state interface

## Known risks

- **Capability.** Following instructions and calling tools reliably needs a model much stronger than a one-million-parameter model trained on a single book. Expected mitigations: a larger and more varied licensed corpus, structured training data for tool-call formats, and keeping decisions that can be rule-based outside the model.
- **Data.** Free, clearly licensed Portuguese text is limited, and every source needs its license recorded.
- **Hardware.** CPU-only training bounds model size and iteration speed. Larger runs may need free or low-cost GPU time.