import json
import json, re
from typing import Tuple
from .base import Evaluator

class AMCEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False

        return correct

    def extract_answer_from_gold_solution(self, solution: str | float):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None

        assert isinstance(completion, str)

        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return None
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]

        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]

        pred = pred.replace(",", "").replace("\n", "")
        is_number, pred = self._is_number(pred)
        if is_number:
            return pred
        else:
            return None


def get_amc_dataset(split):
    with open(f'data/amc/{split}.json') as f:
        examples = json.load(f)

    for ex in examples:
        ex.update(question=ex["problem"] + "\n")
        ex.update(answer=ex["solution"])

    print(f"{len(examples)} {split} examples")
    return examples
