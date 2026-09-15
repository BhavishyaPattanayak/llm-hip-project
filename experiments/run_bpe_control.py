"""Run the separate BPE-control robustness analysis without replacing old results."""
import json
from src.bpe_control import run

if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
