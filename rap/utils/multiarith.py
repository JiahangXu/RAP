import json
import json, re
from typing import Tuple
from .base import Evaluator


def get_multiarith_dataset(split):
    with open(f'data/multiarith/{split}.json') as f:
        examples = json.load(f)

    for ex in examples:
        ex.update(question=ex["problem"].strip() + "\n")
        ex.update(answer=ex["solution"])

    print(f"{len(examples)} {split} examples")
    return examples
