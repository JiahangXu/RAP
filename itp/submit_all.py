import os

scripts = [
    # # gsm8k
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task gsm8k\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task gsm8k\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task gsm8k\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task gsm8k\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task gsm8k\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task gsm8k\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task gsm8k --end_idx 339\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task gsm8k --start_idx 339 --end_idx 666\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task gsm8k --start_idx 666 --end_idx 993\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task gsm8k --start_idx 993\"",
    # gsm8k with limits
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --output_ans_list True --task gsm8k --mcts_rollouts 36 --n_sample_subquestion 10 --end_idx 200 \"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --output_ans_list True --task gsm8k --mcts_rollouts 48 --n_sample_subquestion 10 --end_idx 200 \"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --output_ans_list True --task gsm8k --mcts_rollouts 48 --n_sample_subquestion 10 --end_idx 200 \"",

    # # math
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task math\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task math --end_idx 100\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task math --start_idx 100 --end_idx 200\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task math --start_idx 200 --end_idx 300\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task math --start_idx 300 --end_idx 400\"",
    "\"python run_gsm8k.py --model_type gpt --model_ckpt gpt-35-turbo --data_split test_all --task math --start_idx 400\"",

    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task math --resume 97\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task math --start_idx 300 --end_idx 350 \"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task math --start_idx 350 --end_idx 400 \"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task math --start_idx 400 --end_idx 450 \"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task math --start_idx 450 \"",

    # # math with limits
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10 --resume 36\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",

    # # folio
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task folio\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task folio\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task folio\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task folio\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task folio\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task folio\"",

    # # logiqa
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_sampled --task logiqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_sampled --task logiqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_sampled --task logiqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_sampled --task logiqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_sampled --task logiqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_sampled --task logiqa\"",

    # # svamp
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task svamp\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task svamp\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task svamp\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task svamp\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task svamp\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task svamp\"",
    # # svamp with limits
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --output_ans_list True --task svamp --mcts_rollouts 48 --n_sample_subquestion 10 --end_idx 200 \"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --output_ans_list True --task svamp --mcts_rollouts 48 --n_sample_subquestion 10 --end_idx 200 \"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --output_ans_list True --task svamp --mcts_rollouts 48 --n_sample_subquestion 10 --end_idx 200 \"",


    # # sat
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task sat\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task sat\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task sat\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task sat\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task sat\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task sat --resume 108\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task sat\"",
    
    # # bgqa
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task bgqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task bgqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task bgqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task bgqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task bgqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task bgqa\"",
    
    # multiarith
    
    # # strategyqa
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task strategyqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task strategyqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task strategyqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task strategyqa\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task strategyqa\"",

]

for item in scripts:
    os.system("python submit_genai_a100.py submit --mark RAP-CE --script " + item)
