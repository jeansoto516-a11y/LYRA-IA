import torch


@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_ids: list[int] | None = None,
):
    """Gera tokens um de cada vez (autoregressivo).

    idx: tensor (1, T) com os ids do prompt.
    temperature <= 0 faz a escolha ser sempre o token mais provável (greedy).
    """
    stop = set(stop_ids or [])
    if stop and idx.size(0) != 1:
        raise ValueError("stop_ids só funciona com batch de tamanho 1")

    was_training = model.training
    model.eval()
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth_value = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_value, float("-inf"))

            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                remove = (cumulative - probs) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if stop and next_id.item() in stop:
            break

    model.train(was_training)
    return idx