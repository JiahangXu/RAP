import pickle
import re
from datetime import datetime

from rap.models import QueryLlama, QueryHfModel, QueryVLLM, QueryOpenAI
from rap.utils import get_judge_answer, get_extract_answer_fn, get_one_dataset
from rap.gsm8k_mcts import reasoning_mcts_search

from typing import Tuple
import os
import sys
import torch
import torch.distributed
import torch.backends.cudnn
import fire
import time
import json
import random
# import mlflow
import numpy as np
import math
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# mlflow.autolog()

def load_hf(model_ckpt):
    start_time = time.time()
    print("loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_auth_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model ...")
    model = AutoModelForCausalLM.from_pretrained(model_ckpt, use_auth_token=True, trust_remote_code=True).half().cuda().eval() #! add "half()" to fit in a smaller GPU
    torch.set_default_tensor_type(torch.FloatTensor)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


def load_vllm(model_ckpt):
    from vllm import LLM
    start_time = time.time()
    print("loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)

    print("loading model ...")
    model = LLM(model=model_ckpt, trust_remote_code=True)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


def most_common_element(lst):
    if len(lst) == 0:
        return None
    count = Counter(lst)
    most_common = count.most_common(1)[0]
    return most_common[0]


def main_mcts(model_ckpt='../Llama-2-7b-hf',
              model_type='hf', # choose from ["hf", "vllm", "gpt"]
              task="gsm8k",
              data_split="test_all",
              prompts=None,
              question_prompts=None,
              max_batch_size=2,
              max_response_length=200,
              mcts_rollouts=10,
              n_sample_subquestion=4,
              n_sample_confidence=8,
              temperature=0.8,
              max_depth=6,
              w_exp=1,
              r_alpha=0.5,
              r1_default=1,
              start_idx=0,
              end_idx=math.inf,
              log_dir=None,
              speedup_confidence_batch_size=None,
              output_ans_list=False,
              disable_tqdm=False,
              sing=True):
    if log_dir is None:
        if model_ckpt.endswith("/"):
            model_ckpt = model_ckpt[:-1]
        if sing:
            log_dir = f'/mnt/teamdrive/jiahang/rStar/rap/{task}_mcts_{model_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'
        else:
            log_dir = f'logs/{task}_mcts_{model_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'

    os.makedirs(log_dir, exist_ok=True)
    if output_ans_list:
        if sing:
            log_dir2 = f'/mnt/teamdrive/jiahang/rStar/rap/{task}/{model_ckpt.split("/")[-1]}/'
        else:
            log_dir2 = f'logs/rap/{task}/{model_ckpt.split("/")[-1]}/'
        os.makedirs(log_dir2, exist_ok=True)

    judge_answer = get_judge_answer(task)
    extract_answer_fn = get_extract_answer_fn(task)
    get_dataset = get_one_dataset(task)
    prompts = prompts or f'data/{task}/prompts/interactive_examples.json'
    question_prompts = question_prompts or f'data/{task}/prompts/useful_examples.json'

    # set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model_type == "hf":
        model, tokenizer = load_hf(model_ckpt)
        world_model = QueryHfModel(model, tokenizer, max_response_length=max_response_length, log_file=log_dir)
        eos_token_id = world_model.tokenizer.encode('\n', bos=False, eos=False)[-1]
    elif model_type == "vllm":
        model, tokenizer = load_vllm(model_ckpt)
        world_model = QueryVLLM(model, tokenizer, max_response_length=max_response_length, log_file=log_dir)
        eos_token_id = world_model.tokenizer.encode('\n')[-1]
    elif model_type == "gpt":
        model, tokenizer = model_ckpt, None
        world_model = QueryOpenAI(model, max_response_length=max_response_length, log_file=log_dir)
        eos_token_id = "\n"

    examples = get_dataset(data_split)
    with open(prompts) as f:
        prompts = json.load(f)
    with open(question_prompts) as f:
        question_prompts = json.load(f)

    total_correct = [0] * mcts_rollouts
    total_model_limit_correct, total_major_vote_correct = 0, 0
    for i, example in enumerate((pbar := tqdm(examples, disable=disable_tqdm, position=1))):
        if i < start_idx or i >= end_idx:
            continue
        try:
            question = example['question']
            answer = example['answer']
            trajs, tree, trees, ans_list = reasoning_mcts_search(question, prompts, question_prompts, world_model,
                                                    n_sample_subquestion=n_sample_subquestion,
                                                    mcts_rollouts=mcts_rollouts,
                                                    n_sample_confidence=n_sample_confidence,
                                                    temperature=temperature,
                                                    max_depth=max_depth,
                                                    w_exp=w_exp,
                                                    r_alpha=r_alpha,
                                                    r1_default=r1_default,
                                                    eos_token_id=eos_token_id,
                                                    speedup_confidence_batch_size=speedup_confidence_batch_size,
                                                    task=task,
                                                    extract_answer_fn=extract_answer_fn,
                                                    output_ans_list=output_ans_list,
                                                    random_temp_log=f'{time.time():.6f}.txt',)
            if True: # doesn't test distributed launch
                json_logs = []
                for rollout, traj in enumerate(trajs):
                    output, correct = judge_answer(traj, answer)
                    json_logs.append({
                        'rollout': rollout + 1,
                        'question': question,
                        'answer': answer,
                        'output': output,
                        'correct': correct,
                        'traj': traj,
                    })
                    total_correct[rollout] += correct
                with open(os.path.join(log_dir, f"{example['id']}.json"), 'w') as f:
                    json.dump(json_logs, f, indent=2)
                with open(os.path.join(log_dir, f"{example['id']}.tree"), 'w') as f:
                    f.write(tree)
                with open(os.path.join(log_dir, f"{example['id']}.pkl"), 'wb') as f:
                    pickle.dump(trees, f)

                # Output answer list and model limit/major vote correctness
                if output_ans_list:
                    example_res_sum = {
                        "original_id": example['id'],
                        "sequential_idx": i,
                        "ground_truth": answer,
                        "answer_list": ans_list,
                    }
                    model_limit_correct = False
                    for item in ans_list:
                        if judge_answer(item, answer)[1]:
                            model_limit_correct = True; break
                    major_vote_correct = judge_answer(most_common_element(ans_list), answer)[1]
                    example_res_sum["model_limit_correct"] = model_limit_correct
                    example_res_sum["major_vote_correct"] = major_vote_correct
                    total_model_limit_correct += model_limit_correct
                    total_major_vote_correct += major_vote_correct
                    # import pdb; pdb.set_trace()

                    with open(os.path.join(log_dir2, f"{example['id']}.json"), 'w') as f:
                        json.dump(example_res_sum, f, indent=2)

                tqdm.write(' '.join(f'{c/(i+1-start_idx):0.3f}' for c in total_correct))
                desc = f'acc: {total_correct[-1]}/{i+1-start_idx}={total_correct[-1]/(i+1-start_idx)*100:.3f}'
                if output_ans_list:
                    desc += f", model_limit: {total_model_limit_correct}/{i+1-start_idx}={total_model_limit_correct/(i+1-start_idx)*100:.3f}, " + \
                            f"major_vote: {total_major_vote_correct}/{i+1-start_idx}={total_major_vote_correct/(i+1-start_idx)*100:.3f}"
                pbar.set_description(desc)
                # mlflow.log_metric("acc", total_correct[-1]/(i+1-start_idx)*100)
                # if output_ans_list:
                #     mlflow.log_metric("model_limit_acc", total_model_limit_correct/(i+1-start_idx)*100)
                #     mlflow.log_metric("major_vote_acc", total_major_vote_correct/(i+1-start_idx)*100)
        except Exception as e:
            print(f"Error in example {example['id']}: {e}")

if __name__ == '__main__':
    fire.Fire(main_mcts)
