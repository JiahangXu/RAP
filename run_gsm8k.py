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
        log_dir = f'logs/gsm8k_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'
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
        trajs, tree, trees = reasoning_mcts_search(question, prompts, question_prompts,
                                                   n_sample_subquestion=n_sample_subquestion,
                                                   mcts_rollouts=mcts_rollouts,
                                                   n_sample_confidence=n_sample_confidence,
                                                   temperature=temperature,
                                                   max_depth=max_depth,
                                                   w_exp=w_exp,
                                                   r_alpha=r_alpha,
                                                   r1_default=r1_default,
                                                   speedup_confidence_batch_size=speedup_confidence_batch_size)
        
        if True:
            json_logs = []
            for rollout, traj in enumerate(trajs):
                output, correct = judge_answer_gsm8k(traj, answer)
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
            print(f'{total_correct[-1]}/{i+1-resume}={total_correct[-1]/(i+1-resume):.2f}')

        break


if __name__ == '__main__':
    fire.Fire(main_mcts)

'''
nohup python run_gsm8k.py --part 0 > gpt4_gsm8k_part0.txt  2>&1 &
nohup python run_gsm8k.py --part 1 > gpt4_gsm8k_part1.txt  2>&1 &
nohup python run_gsm8k.py --part 2 > gpt4_gsm8k_part2.txt  2>&1 &
nohup python run_gsm8k.py --part 3 > gpt4_gsm8k_part3.txt  2>&1 &
nohup python run_gsm8k.py --part 4 > gpt4_gsm8k_part4.txt  2>&1 &
nohup python run_gsm8k.py --part 0 > gpt3.5_gsm8k_part0.txt  2>&1 &
nohup python run_gsm8k.py --part 1 > gpt3.5_gsm8k_part1.txt  2>&1 &
nohup python run_gsm8k.py --part 2 > gpt3.5_gsm8k_part2.txt  2>&1 &
nohup python run_gsm8k.py --part 3 > gpt3.5_gsm8k_part3.txt  2>&1 &
nohup python run_gsm8k.py --part 4 > gpt3.5_gsm8k_part4.txt  2>&1 &


nohup python run_gsm8k.py --part 0 --mcts_rollouts 5 > gpt4_gsm8k_rollout5_part0.txt  2>&1 &[1] 32584
nohup python run_gsm8k.py --part 1 --mcts_rollouts 5 > gpt4_gsm8k_rollout5_part1.txt  2>&1 &[2] 32726
nohup python run_gsm8k.py --part 2 --mcts_rollouts 5 > gpt4_gsm8k_rollout5_part2.txt  2>&1 &[3] 32832
nohup python run_gsm8k.py --part 3 --mcts_rollouts 5 > gpt4_gsm8k_rollout5_part3.txt  2>&1 &[4] 32968
nohup python run_gsm8k.py --part 4 --mcts_rollouts 5 > gpt4_gsm8k_rollout5_part4.txt  2>&1 &[5] 33091

nohup python run_gsm8k.py --part 0 --mcts_rollouts 5 > gpt3.5_gsm8k_rollout5_part0.txt  2>&1 &
nohup python run_gsm8k.py --part 1 --mcts_rollouts 5 > gpt3.5_gsm8k_rollout5_part1.txt  2>&1 &
nohup python run_gsm8k.py --part 2 --mcts_rollouts 5 > gpt3.5_gsm8k_rollout5_part2.txt  2>&1 &
nohup python run_gsm8k.py --part 3 --mcts_rollouts 5 > gpt3.5_gsm8k_rollout5_part3.txt  2>&1 &
nohup python run_gsm8k.py --part 4 --mcts_rollouts 5 > gpt3.5_gsm8k_rollout5_part4.txt  2>&1 &



'''