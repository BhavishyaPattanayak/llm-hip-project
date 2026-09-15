"""Story-context surprisal with exact reading-region alignment."""
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ("gpt2", "distilgpt2")


def load_model(model_name, device="auto", local_files_only=False):
    if model_name not in MODELS:
        raise ValueError(f"Choose from {MODELS}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device).eval()
    return tokenizer, model, device


def encode_regions(words, tokenizer):
    """Tokenize a story once; assign offsets by overlap with known region spans.

    Separator spaces belong to no region. A BPE covering a leading separator
    and the following word therefore belongs exclusively to that word.
    """
    if not words or any(not w or any(c.isspace() for c in w) for w in words):
        raise ValueError("Expected nonempty reading regions without whitespace")
    text = " ".join(words)
    spans = []
    start = 0
    for word in words:
        spans.append((start, start + len(word)))
        start += len(word) + 1
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                        truncation=False, verbose=False)
    owners = []
    region = 0
    for start, end in encoded["offset_mapping"]:
        while region < len(spans) and spans[region][1] <= start:
            region += 1
        if (region == len(spans) or end <= spans[region][0] or start == end
                or (region + 1 < len(spans) and end > spans[region + 1][0])):
            raise ValueError(f"BPE offset {(start, end)} has ambiguous region ownership")
        owners.append(region)
    if set(owners) != set(range(len(words))):
        raise ValueError("Some reading regions have no BPE tokens")
    return encoded["input_ids"], owners


def score_tokens(ids, model, device, stride=256):
    """Score each token once using logits immediately before its position.

    Initial window uses the available prefix. Subsequent windows score only
    their final stride targets, retaining at least window-stride predecessors.
    Positions restart at zero in each window. No cross-story context or BOS.
    """
    window = model.config.max_position_embeddings
    if not 1 <= stride < window:
        raise ValueError("stride must be between 1 and context window minus 1")
    scores = [float("nan")] * len(ids)
    target_start = 1
    with torch.inference_mode():
        while target_start < len(ids):
            end = min(len(ids), window if target_start == 1 else target_start + stride)
            begin = max(0, end - window)
            x = torch.tensor([ids[begin:end]], dtype=torch.long, device=device)
            logits = model(input_ids=x, use_cache=False).logits[0]
            local_start = target_start - begin
            loss = torch.nn.functional.cross_entropy(
                logits[local_start - 1:-1].float(), x[0, local_start:], reduction="none")
            scores[target_start:end] = (loss / math.log(2)).cpu().tolist()
            target_start = end
    return scores


def score_story(rows, tokenizer, model, device, stride=256):
    ids, owners = encode_regions([r["word"] for r in rows], tokenizer)
    scores = score_tokens(ids, model, device, stride)
    sums = [0.0] * len(rows)
    counts = [0] * len(rows)
    for owner, score in zip(owners, scores):
        counts[owner] += 1
        sums[owner] += score
    # Never report a partial first-word surprisal, even for a multi-BPE word.
    sums[0] = float("nan")
    return [dict(row, surprisal=sums[i], n_bpe=counts[i]) for i, row in enumerate(rows)]
