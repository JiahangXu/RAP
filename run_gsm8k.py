import pickle
import re
from datetime import datetime
from rap.models import QueryLlama
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
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024


def setup_logging(log_dir, llama_ckpt):
    if log_dir is None:
        log_dir = f'logs/gsm8k_mcts_{llama_ckpt.split("/")[-1]}/{datetime.now().strftime("%Y-%m%d-%H%M")}'
    os.makedirs(log_dir, exist_ok=True)   
    return log_dir


def setup_random():
    # set random seed 
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_hf(llama_ckpt):
    print("loading tokenizer ...")
    start_time = time.time()
    tokenizer = LlamaTokenizer.from_pretrained(llama_ckpt, use_auth_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading model ...")
    model = LlamaForCausalLM.from_pretrained(llama_ckpt, use_auth_token=True).half().cuda().eval() #! add "half()" to fit in a smaller GPU
    torch.set_default_tensor_type(torch.FloatTensor)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model, tokenizer


def main_mcts(llama_ckpt='meta-llama/Llama-2-7b-hf',
              prompts='data/gsm8k/prompts/interactive_examples.json',
              question_prompts='data/gsm8k/prompts/useful_examples.json',
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
              resume=0,
              log_dir=None,
              speedup_confidence_batch_size=None):
    setup_random()
    log_dir = setup_logging(log_dir, llama_ckpt)

    model, tokenizer = load_hf(llama_ckpt)

    world_model = QueryLlama(model, tokenizer, max_response_length=max_response_length, log_file=None)

    examples = get_gsm8k_dataset('test')
    with open(prompts) as f:
        prompts = json.load(f)
    with open(question_prompts) as f:
        question_prompts = json.load(f)

    total_correct = [0] * mcts_rollouts
    for i, example in enumerate((pbar := tqdm(examples, disable=True, position=1))):
        if i < resume:
            continue
        question = example['question']
        answer = example['answer']
        answer = re.search('#### .*?([ $.0-9,\\-]+)', answer)
        answer = '' if answer is None else answer[1].replace(',', '').replace(' ', '').replace('$', '')
        trajs, tree, trees, extra_info = reasoning_mcts_search(question, prompts, question_prompts, world_model,
                                                   n_sample_subquestion=n_sample_subquestion,
                                                   mcts_rollouts=mcts_rollouts,
                                                   n_sample_confidence=n_sample_confidence,
                                                   temperature=temperature,
                                                   max_depth=max_depth,
                                                   w_exp=w_exp,
                                                   r_alpha=r_alpha,
                                                   r1_default=r1_default,
                                                   eos_token_id=world_model.tokenizer.encode('\n')[-1],
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
                    'query_LM_counter': extra_info.query_LM_counter,
                    'num_hit_max_depth': extra_info.num_hit_max_depth,
                    'exec_time': extra_info.exec_time
                })
                total_correct[rollout] += correct
            with open(os.path.join(log_dir, f'{i:04d}.json'), 'w') as f:
                json.dump(json_logs, f, indent=2)
            with open(os.path.join(log_dir, f'{i:04d}.tree'), 'w') as f:
                f.write(tree)
            with open(os.path.join(log_dir, f'{i:04d}.pkl'), 'wb') as f:
                pickle.dump(trees, f)
            tqdm.write(' '.join(f'{c/(i+1-resume):0.3f}' for c in total_correct))
            pbar.set_description(f'{total_correct[-1]}/{i+1-resume}={total_correct[-1]/(i+1-resume):.2f}')
        break


if __name__ == '__main__':
    fire.Fire(main_mcts)
