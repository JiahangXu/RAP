import json

def get_gsm8khard_dataset(split):
    with open(f'data/gsm8khard/{split}.json') as f:
        examples = json.load(f)

    for ex in examples:
        ex.update(question=ex["problem"] + "\n")
        ex.update(answer=ex["solution"])

    print(f"{len(examples)} {split} examples")
    return examples