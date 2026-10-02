# Model

## Architecture

Lyra is a decoder-only causal Transformer written from scratch in PyTorch.

| Component | Choice |
|-----------|--------|
| Attention | Multi-head causal self-attention (PyTorch `scaled_dot_product_attention`, `is_causal=True`) |
| Normalization | RMSNorm, pre-norm placement |
| Feed-forward | Linear, GELU, Linear (no biases) |
| Positional information | Learned absolute position embeddings |
| Output head | Tied with the token embedding matrix |
| Regularization | Dropout (0.1 by default) |
| Residual connections | Around both attention and feed-forward in every block |

## Default configuration (Lyra 0.1)

| Hyperparameter | Value |
|----------------|-------|
| Vocabulary size | 1,500 |
| Context length | 128 tokens |
| Model dimension | 128 |
| Layers | 4 |
| Attention heads | 4 (head dimension 32) |
| Feed-forward dimension | 512 |

## Parameter count

Total: **995,968 parameters** (about 1.0M, roughly 4 MB in fp32).

| Part | Parameters |
|------|-----------:|
| Token embeddings (shared with output head) | 192,000 |
| Position embeddings | 16,384 |
| Transformer blocks (4 x 196,864) | 787,456 |
| Final normalization | 128 |

Training state (AdamW moments) adds about two more copies of the weights, so a full checkpoint is roughly 12 MB.

## Tokenizer

Byte-level BPE implemented in `core/tokenizer/bpe.py`.

- Every text is encoded as UTF-8 bytes, so no input can be out of vocabulary.
- Three special tokens: `<pad>`, `<bos>`, `<eos>`; then 256 byte tokens; then learned merges (1,241 merges for a vocabulary of 1,500).
- A regular-expression pre-tokenizer splits text into words (with the leading space), numbers and punctuation before merging.
- The tokenizer is saved and loaded independently of the model (JSON).

## Generation

Autoregressive sampling in `core/generation/sampling.py`:

- temperature (a value of 0 or lower selects the most likely token),
- top-k and top-p (nucleus) filtering,
- maximum number of new tokens,
- stop tokens (batch size 1).

## Limitations

- Context is 128 tokens, about 360 characters of Portuguese with the current tokenizer. Learned positions cannot be extended beyond the trained context.
- No KV cache yet: every generated token recomputes attention over the whole window.
- The vocabulary is small and was trained on a single book, so uncommon words are split into many pieces.
- Outputs follow the style of the training corpus but are not coherent beyond a short phrase.

## Growth path

Model size is controlled entirely by `LyraConfig` (`d_model`, `n_layers`, `n_heads`, `d_ff`, `context_length`, `vocab_size`), so larger models need no code changes. Candidate future changes, none implemented yet: rotary position embeddings, a KV cache for inference, a larger tokenizer vocabulary trained on a bigger corpus.