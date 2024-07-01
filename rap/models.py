from abc import ABC, abstractmethod
from typing import List

import os
import torch
from vllm import SamplingParams


class QueryLM(ABC):
    @abstractmethod
    def query_LM(self, prompt, **gen_kwargs):
        pass

    @abstractmethod
    def query_next_token(self, prompt: List[str]):
        pass


class QueryVLLM(QueryLM):
    def __init__(self, model, tokenizer, max_response_length, log_file) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = 1 # only test batch_size=1
        self.yes_no = tokenizer.encode('Yes No')[1:]

    @torch.no_grad()
    def query_next_token(self, prompt):
        sampling_params = SamplingParams(logprobs=5, max_tokens=1)
        completions = self.model.generate(prompt, sampling_params)
        ret = []
        for completion in completions:
            logits = completion.outputs[0].logprobs[0]
            yes_probs = logits.get(self.yes_no[0], None).logprob if logits.get(self.yes_no[0], None) else -float("inf")
            no_probs = logits.get(self.yes_no[1], None).logprob if logits.get(self.yes_no[1], None) else -float("inf")
            ret.append(torch.tensor([yes_probs, no_probs]).cuda().float())
        ret = torch.stack(ret, dim=0)
        dist = torch.softmax(ret, dim=-1)
        return dist

    def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
        sampling_params = SamplingParams(temperature=temperature, n=num_return_sequences, max_tokens=self.max_response_length, stop_token_ids=[eos_token_id], stop="\n")
        completions = self.model.generate(prompt, sampling_params)
        all_results = [prompt + output.text + ("\n" if "\n" not in output.text else "") for output in completions[0].outputs]

        if self.log_file:
            with open(os.path.join(self.log_file, "token_num.txt"), "a") as f:
                all_tokens = [len(output.token_ids) for output in completions[0].outputs]
                f.write(",".join([str(tokens) for tokens in all_tokens]) + "\n")
        return all_results

class QueryHfModel(QueryLM):
    @torch.no_grad()
    def query_next_token(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        ret = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            tokens = torch.tensor([tokens]).cuda().long()
            output = self.model.forward(tokens).logits[:, -1, :]
            ret.append(output)
        outputs = torch.cat(ret, dim=0)
        filtered = outputs[:, self.yes_no]
        dist = torch.softmax(filtered, dim=-1)
        return dist

    def __init__(self, model, tokenizer, max_response_length, log_file) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = 1 # only test batch_size=1
        self.yes_no = self.tokenizer.encode('Yes No')[1:]

    @torch.no_grad()
    def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
        temperature = temperature if do_sample else 0
        all_results = []
        for start in range(0, num_return_sequences, self.max_batch_size):
            end = min(start + self.max_batch_size, num_return_sequences)
            encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(self.model.device)
            results = self.model.generate(encoded_prompt, max_length=self.max_response_length+len(encoded_prompt[0]), temperature=temperature, num_return_sequences=end - start, pad_token_id=eos_token_id, eos_token_id=eos_token_id, do_sample=do_sample)

            if len(results.shape) > 2:
                results.squeeze_()
            for generated_sequence in results:
                generated_sequence = generated_sequence.tolist()
                # Decode text
                text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                all_results.append(text)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("="*50+"\n")
                f.write(prompt + "\n")
                for result in all_results:
                    f.write("-"*50+"\n")
                    f.write(result.replace(prompt, "") + "\n")
        return all_results


class QueryLlama(QueryLM):
    def __init__(self, llamamodel, max_response_length, log_file) -> None:
        self.llamamodel = llamamodel
        self.tokenizer = self.llamamodel.tokenizer
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = llamamodel.model.params.max_batch_size
        self.yes_no = self.tokenizer.encode('Yes No', bos=False, eos=False)

    def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
        temperature = temperature if do_sample else 0
        all_results = []
        for start in range(0, num_return_sequences, self.max_batch_size):
            end = min(start + self.max_batch_size, num_return_sequences)
            results = self.llamamodel.generate([prompt] * (end - start), max_gen_len=self.max_response_length, temperature=temperature, eos_token_id=eos_token_id)
            all_results.extend(results)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("="*50+"\n")
                f.write(prompt + "\n")
                for result in all_results:
                    f.write("-"*50+"\n")
                    f.write(result.replace(prompt, "") + "\n")
        return all_results

    @torch.no_grad()
    def query_next_token(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        ret = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
            tokens = torch.tensor([tokens]).cuda().long()
            output, h = self.llamamodel.model.forward(tokens, start_pos=0)
            ret.append(output)
        outputs = torch.cat(ret, dim=0)
        filtered = outputs[:, self.yes_no]
        dist = torch.softmax(filtered, dim=-1)
        return dist

