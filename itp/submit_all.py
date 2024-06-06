import os

scripts = [
    # # gsm8k
    # nohup python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all > gsm8k_llama2_7b.txt 2>&1 &
    # nohup python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all > gsm8k_llama2_13b.txt 2>&1 &
    # nohup python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all > gsm8k_llama3_8b_instruct.txt 2>&1 &
    # nohup python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all > gsm8k_phi3_mini.txt 2>&1 &
    # nohup python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all > gsm8k_mistral_7b.txt 2>&1 &
    # nohup python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all > gsm8k_llama3_8b.txt 2>&1 &

    # # math
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task math\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task math\"",
    # # math with limits
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
    # "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_sampled --output_ans_list True --task math --mcts_rollouts 32 --n_sample_subquestion 10\"",
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

    # svamp
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task svamp\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task svamp\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task svamp\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task svamp\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task svamp\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task svamp\"",

    # sat
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-7b-hf --data_split test_all --task sat\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Llama-2-13b-hf --data_split test_all --task sat\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B-Instruct --data_split test_all --task sat\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Phi-3-mini-4k-instruct --data_split test_all --task sat\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Mistral-7B-v0.1 --data_split test_all --task sat\"",
    "\"python run_gsm8k.py --model_type vllm --model_ckpt ../Meta-Llama-3-8B --data_split test_all --task sat\"",
]

for item in scripts:
    os.system("python submit_genai_a100.py submit --mark RAP_limit --script " + item)
