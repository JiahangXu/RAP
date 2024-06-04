from .gsm8k import GSM8KEvaluator, get_gsm8k_dataset
from .folio import FOLIOEvaluator
from .math import MATHEvaluator, get_math_dataset
from .logiqa import LOGIQAEvaluator

def get_evaluator(task):
    if task == 'gsm8k':
        return GSM8KEvaluator()
    elif task == 'folio':
        return FOLIOEvaluator()
    elif task == 'math':
        return MATHEvaluator()
    elif task == 'logiqa':
        return LOGIQAEvaluator()

def get_judge_answer(task):
    evaluator = get_evaluator(task)
    return evaluator.judge_answer

def get_extract_answer_fn(task):
    evaluator = get_evaluator(task)
    return evaluator.extract_answer_from_model_completion

def get_one_dataset(task):
    if task == 'gsm8k':
        return get_gsm8k_dataset
    elif task == 'math':
        return get_math_dataset
    # if task == 'gsm8k':
    #     evaluator = GSM8KEvaluator()
    #     return evaluator.judge_answer