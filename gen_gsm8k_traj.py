import pickle
import re
from datetime import datetime

# from rap.models import QueryLlama
from rap.utils.gsm8k import judge_answer_gsm8k, get_gsm8k_dataset
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
import numpy as np
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def main_mcts(llama_ckpt='gpt3.5',
              prompts='data/gsm8k/prompts/interactive_examples.json',
              question_prompts='data/gsm8k/prompts/useful_examples.json',
              max_batch_size=2,
              max_response_length=200,
              mcts_rollouts=1,
              n_sample_subquestion=4,
              n_sample_confidence=8,
              temperature=0.8,
              max_depth=6,
              w_exp=1,
              r_alpha=0.5,
              r1_default=1,
              resume=0,
              log_dir=None,
              speedup_confidence_batch_size=None,
              part=0):
    if log_dir is None:
        log_dir = f'logs/gsm8k_train_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'
    os.makedirs(log_dir, exist_ok=True)

    # set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    split = "train"
    if split == "train":
        piece_size = 500
        all_size = 7473
    else: # split = "test"
        piece_size = 264
        all_size = 1319
    examples = get_gsm8k_dataset(split)
    with open(prompts) as f:
        prompts = json.load(f)
    with open(question_prompts) as f:
        question_prompts = json.load(f)

    print(f"Part {part}, from example {part * piece_size} to example {min((part + 1) * piece_size, all_size)}")
    total_correct = [0] * mcts_rollouts
    for i, example in enumerate((pbar := tqdm(examples[part * piece_size: min((part + 1) * piece_size, all_size)], disable=True, position=1))):
        example_id = i + part * piece_size
        if example_id < resume:
            continue
        question = example['question']
        answer = example['answer']
        answer = re.search('#### .*?([ $.0-9,\\-]+)', answer)
        answer = '' if answer is None else answer[1].replace(',', '').replace(' ', '').replace('$', '')
        
        count = 0
        correct = False
        while count <= 5 and not correct:
            print(f"============= {count}-th trial for example {example_id} =============")
            trajs, tree, trees = reasoning_mcts_search(question, prompts, question_prompts,
                                                    n_sample_subquestion=n_sample_subquestion,
                                                    mcts_rollouts=1,
                                                    n_sample_confidence=n_sample_confidence,
                                                    temperature=temperature,
                                                    max_depth=max_depth,
                                                    w_exp=w_exp,
                                                    r_alpha=r_alpha,
                                                    r1_default=r1_default,
                                                    speedup_confidence_batch_size=speedup_confidence_batch_size)
        
            count += 1
            rollout = 0
            traj = trajs[0]
            json_logs = []
            output, correct = judge_answer_gsm8k(traj, answer)
            if correct:
                json_logs.append({
                    'rollout': rollout + 1,
                    'question': question,
                    'answer': answer,
                    'output': output,
                    'correct': correct,
                    'traj': traj,
                })
                total_correct[rollout] += correct
                with open(os.path.join(log_dir, f'{example_id:04d}.json'), 'w') as f:
                    json.dump(json_logs, f, indent=2)
                with open(os.path.join(log_dir, f'{example_id:04d}.tree'), 'w') as f:
                    f.write(tree)
                with open(os.path.join(log_dir, f'{example_id:04d}.pkl'), 'wb') as f:
                    pickle.dump(trees, f)
                tqdm.write(' '.join(f'{c/(i+1-resume):0.3f}' for c in total_correct))
                pbar.set_description(f'{total_correct[-1]}/{i+1-resume}={total_correct[-1]/(i+1-resume):.2f}')
                # print(f'{total_correct[-1]}/{i+1-resume}={total_correct[-1]/(i+1-resume):.2f}')
                print(f"Example {example_id} get correct in {count}-th trial, save and break.")
                continue
            elif count < 5:
                print(f"Example {example_id} get wrong in {count}-th trial. Try again")

            if count == 5 and not correct:
                print(f"Example {example_id} still get wrong after {count} trial. Skip it.")


if __name__ == '__main__':
    fire.Fire(main_mcts)

'''
nohup python gen_gsm8k_traj.py --part 0 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part0.txt  2>&1 &[1] 51603
nohup python gen_gsm8k_traj.py --part 1 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part1.txt  2>&1 &[2] 52668
nohup python gen_gsm8k_traj.py --part 2 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part2.txt  2>&1 &[3] 52858
nohup python gen_gsm8k_traj.py --part 3 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part3.txt  2>&1 &[4] 53117
nohup python gen_gsm8k_traj.py --part 4 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part4.txt  2>&1 &[5] 53248
nohup python gen_gsm8k_traj.py --part 5 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part5.txt  2>&1 &[6] 53455
nohup python gen_gsm8k_traj.py --part 6 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part6.txt  2>&1 &[7] 53568
nohup python gen_gsm8k_traj.py --part 7 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part7.txt  2>&1 &[8] 53788
nohup python gen_gsm8k_traj.py --part 8 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part8.txt  2>&1 &[9] 53937
nohup python gen_gsm8k_traj.py --part 9 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part9.txt  2>&1 &[10] 54073
nohup python gen_gsm8k_traj.py --part 10 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part10.txt  2>&1 &[11] 54383
nohup python gen_gsm8k_traj.py --part 11 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part11.txt  2>&1 &[12] 54526
nohup python gen_gsm8k_traj.py --part 12 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part12.txt  2>&1 &[13] 54736
nohup python gen_gsm8k_traj.py --part 13 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part13.txt  2>&1 &[14] 54970
nohup python gen_gsm8k_traj.py --part 14 > logs/gsm8k_train_mcts_gpt3.5/gpt3.5_gsm8k_train_part14.txt  2>&1 &[15] 55239


'''