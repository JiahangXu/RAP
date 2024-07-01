from abc import ABC, abstractmethod
import os
import time
import torch
from tqdm import tqdm
import concurrent.futures
from vllm import SamplingParams
from typing import List
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.environ.get('GPT35_KEY', ''),
    api_version="2023-12-01-preview",
    azure_endpoint="https://gcrendpoint.azurewebsites.net",
)

class QueryLM(ABC):
    @abstractmethod
    def query_LM(self, prompt, **gen_kwargs):
        pass

    @abstractmethod
    def query_next_token(self, prompt: List[str]):
        pass


class QueryOpenAI(QueryLM):
    def __init__(self, model_ckpt, max_response_length, log_file) -> None:
        self.model_ckpt = model_ckpt
        self.max_response_length = max_response_length
        self.log_file = log_file
        self.max_batch_size = 1

    def generate_with_OpenAI_model(
        self,
        prompt,
        max_tokens=256,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        stop=["\n"],
    ):
        messages = [{"role": "user", "content": prompt}]
        parameters = {
            "model": self.model_ckpt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            "seed": 1,
        }

        ans, timeout = "", 5
        while not ans:
            try:
                time.sleep(timeout)
                completion = client.chat.completions.create(messages=messages, **parameters)
                ans = completion.choices[0].message.content
            except Exception as e:
                print(e)
            if not ans:
                timeout = timeout * 2
                if timeout > 120:
                    timeout = 1
                try:
                    print(f"Will retry after {timeout} seconds ...")
                except:
                    pass
        return ans, completion.usage.completion_tokens

    def generate_n_with_OpenAI_model(
        self,
        prompt,
        n=1,
        max_tokens=256,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        stop=["\n"],
        max_threads=32,
        disable_tqdm=True,
    ):
        preds = []
        completion_tokens = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(self.generate_with_OpenAI_model, prompt, max_tokens, temperature, top_k, top_p, stop) for _ in range(n)]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running evaluate', disable=disable_tqdm):
                ans, c_tokens = future.result()
                preds.append(ans)
                completion_tokens.append(c_tokens)
        return preds, completion_tokens

    def query_next_token(self, prompt):
        ret = []
        for p in prompt:
            completion, _ = self.generate_with_OpenAI_model(p, temperature=0.0, max_tokens=1)
            if completion.lower().startswith("yes"):
                ret.append(torch.tensor([1, 0]).cuda().float())
            else:
                ret.append(torch.tensor([0, 1]).cuda().float())
        ret = torch.stack(ret, dim=0)
        dist = torch.softmax(ret, dim=-1)
        return dist

    def query_LM(self, prompt, eos_token_id, num_return_sequences=1, do_sample=True, temperature=0.8):
        completions, c_tokens = self.generate_n_with_OpenAI_model(
            prompt,
            n=num_return_sequences,
            max_tokens=self.max_response_length,
            temperature=temperature,
            stop=[eos_token_id],)
        
        all_results = [prompt + output + ("\n" if "\n" not in output else "") for output in completions]

        if self.log_file:
            with open(os.path.join(self.log_file, "token_num.txt"), "a") as f:
                f.write(",".join([str(tokens) for tokens in c_tokens]) + "\n")
        return all_results

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

