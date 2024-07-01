from .gsm8k import GSM8KEvaluator, get_gsm8k_dataset
from .folio import FOLIOEvaluator, get_folio_dataset
from .math import MATHEvaluator, get_math_dataset
from .logiqa import LOGIQAEvaluator, get_logiqa_dataset
from .sat import SATEvaluator, get_sat_dataset
from .svamp import SVAMPEvaluator, get_svamp_dataset
from .bgqa import get_bgqa_dataset
from .strategyqa import STGEvaluator, get_strategyqa_dataset

def get_evaluator(task):
    if task == 'gsm8k':
        return GSM8KEvaluator()
    elif task == 'folio' or task == 'bgqa':
        return FOLIOEvaluator()
    elif task == 'math':
        return MATHEvaluator()
    elif task == 'logiqa':
        return LOGIQAEvaluator()
    elif task == 'sat':
        return SATEvaluator()
    elif task == 'svamp':
        return SVAMPEvaluator()
    elif task == 'strategyqa':
        return STGEvaluator()

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
    elif task == "logiqa":
        return get_logiqa_dataset
    elif task == 'folio':
        return get_folio_dataset
    elif task == 'sat':
        return get_sat_dataset
    elif task == 'svamp':
        return get_svamp_dataset
    elif task == 'bgqa':
        return get_bgqa_dataset
    elif task == 'strategyqa':
        return get_strategyqa_dataset
    else:
        raise NotImplementedError(f"Task {task} not implemented yet")
