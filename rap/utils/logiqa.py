import json
from .base import Evaluator

class LOGIQAEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
    
    def _format_answer(self, answer: str):
        assert isinstance(answer, str)
        
        if answer.lower() in ['a', 'a)', '(a)']:
            return 'a'
        elif answer.lower() in ['b', 'b)', '(b)']:
            return 'b'
        elif answer.lower() in ['c', 'c)', '(c)']:
            return 'c'
        elif answer.lower() in ['d', 'd)', '(d)']:
            return 'd'
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
        answer_split = self.isolate_answer(completion)    # note that it is lower case a-d
        try:
            if "a" in answer_split:
                assert not any([choice in answer_split for choice in ["b", "c", "d"]])
                return "a"
            elif "b" in answer_split:
                assert not any([choice in answer_split for choice in ["a", "c", "d"]])
                return "b"
            elif "c" in answer_split:
                assert not any([choice in answer_split for choice in ["a", "b", "d"]])
                return "c"
            elif "d" in answer_split:
                assert not any([choice in answer_split for choice in ["a", "b", "c"]])
                return "d"
            else:
                raise ValueError
        except:
            return None
    

def get_logiqa_dataset(split):
    with open(f'data/logiqa/{split}.json') as f:
        examples = json.load(f)

    for ex in examples:
        ex.update(question=ex["problem"] + "\n")
        ex.update(answer=ex["solution"])

    print(f"{len(examples)} {split} examples")
    return examples
