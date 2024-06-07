import json
def get_sat_dataset(split="test_all"):
    with open(f'data/sat/{split}.json') as f:
        examples = json.load(f)

    for ex in examples:
        prob_list = ex["problem"].split("\n")
        if prob_list[0].endswith(" ?$"):
            prob_list[0] = prob_list[0][:-3] + "$ ?"
            question = " ".join(prob_list)
        else:
            question = ex["problem"].replace("\n", " ")
        question = question.replace("A. ", "A.").replace("B. ", "B.").replace("C. ", "C.").replace("D. ", "D.")
        ex.update(problem=question + "\n")

    print(f"{len(examples)} {split} examples")
    return examples

a = get_sat_dataset()
with open("data/sat/test_all_1.json", "w") as f:
    json.dump(a, f, indent=4)