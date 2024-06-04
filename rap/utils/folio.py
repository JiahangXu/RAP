import json
from .base import Evaluator

class FOLIOEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
    
    def _format_answer(self, answer: str):
        if answer.lower() in ['true', 'yes', 'correct', 'positive', 'affirmative', 'right', '1', 't', 'y']:
            return 'true'
        elif answer.lower() in ['false', 'no', 'incorrect', 'negative', 'wrong', '0', 'f', 'n']:
            return 'false'
        else:
            return answer.lower()
    
    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False
        
        return self._format_answer(answer_a) == self._format_answer(answer_b)
    
    def extract_answer_from_gold_solution(self, solution: str):
        if solution is None:
            return None

        return self._format_answer(solution)
    
    def extract_answer_from_model_completion(self, completion: str):
        answer = self.isolate_answer(completion)
        if answer is None:
            return None
    
        return self._format_answer(answer) 


def get_folio_dataset(split):
    with open(f'data/folio/{split}.json') as f:
        examples = json.load(f)

    for ex in examples:
        ex.update(question=ex["problem"] + "\n")
        ex.update(answer=FOLIOEvaluator().extract_answer_from_gold_solution(ex["solution"]))

    print(f"{len(examples)} {split} examples")
    return examples
