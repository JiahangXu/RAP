from rap.utils.toolkit_for_MATH.latex_answer_check import latex_answer_check
import re
import json
from .base import Evaluator

class SATEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
    
        self.alphas = ('A', 'B', 'C', 'D', 'E', 'F', 'a', 'b', 'c', 'd', 'e', 'f')
    
    def latex_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return False
        
        if answer_a == "" or answer_b == "":
            return False
        
        return latex_answer_check(
            answer_a,
            answer_b,
            split=None,
            extract_policy="flex",
            eval_policy="aggressive",
        )
    
    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a in self.alphas and answer_b in self.alphas:
            return answer_a.upper() == answer_b.upper()
        elif answer_a not in self.alphas and answer_b not in self.alphas:
            return self.latex_answers_equiv(answer_a, answer_b)
        else:
            return self.latex_answers_equiv(answer_a, answer_b)
        

    def extract_answer_from_model_completion(self, completion: str):
        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False 
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = re.findall(r'A|B|C|D|E|F', pred)

        if len(pred) == 0:
            return "C"
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]
    
        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]
        
        if pred:
            return pred
        else:
            return "C"
    
    def extract_answer_from_gold_solution(self, solution: str):
        """Extract the answer from the gold solution."""
        return solution


def get_sat_dataset(split):
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
        ex.update(question=question + "\n")

        ex.update(answer=ex["solution"])

    print(f"{len(examples)} {split} examples")
    return examples
