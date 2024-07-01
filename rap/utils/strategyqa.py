import re
import json
from .base import Evaluator
from fuzzywuzzy import fuzz, process

class STGEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _format_answer(self, answer: str):
        if answer.lower() in ["proved", "true", "yes", "correct", "positive", "affirmative", "right", "1", "t", "y"]:
            return "true"
        elif answer.lower() in ["disproved", "false", "no", "incorrect", "negative", "wrong", "0", "f", "n"]:
            return "false"
        else:
            return answer.lower()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False

        assert isinstance(answer_a, str) and isinstance(answer_b, str)

        format_answer_a = self._format_answer(answer_a)
        format_answer_b = self._format_answer(answer_b)
        return format_answer_a == format_answer_b or fuzz.token_sort_ratio(format_answer_a, format_answer_b) >= 90

    def extract_answer_from_gold_solution(self, solution: str):
        if solution is None:
            return None

        assert isinstance(solution, str)

        return self._format_answer(solution)

    def extract_answer_from_model_completion(self, completion: str):
        if completion is None:
            return None

        assert isinstance(completion, str)

        answer = self.isolate_answer(completion)
        if answer is None:
            return None

        return self._format_answer(answer)


def get_strategyqa_dataset(split):
    with open(f'data/strategyqa/{split}.json') as f:
        examples = json.load(f)

    for ex in examples:
        ex.update(question=ex["problem"] + "\n")
        ex.update(answer=ex["solution"])

    print(f"{len(examples)} {split} examples")
    return examples
