"""Manual inspection of exact region-to-BPE mapping (no inference)."""
from transformers import AutoTokenizer
from src.surprisal import encode_regions


def main():
    words = ['The', 'extraordinarily', 'beautiful', 'butterfly', 'disappeared.']
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    ids, owners = encode_regions(words, tokenizer)
    for token, owner in zip(ids, owners):
        print(owner, repr(words[owner]), repr(tokenizer.decode([token])))


if __name__ == '__main__':
    main()
