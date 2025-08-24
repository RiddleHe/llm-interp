import torch
import numpy as np 
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import heapq
import argparse
import openai
from typing import List, Dict, Any
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=1024)
args = parser.parse_args()

budget = args.budget
max_tokens = budget * 2

prompt = """We can scramble a string s to get a string t using the following algorithm:

If the length of the string is 1, stop.
If the length of the string is > 1, do the following:
Split the string into two non-empty substrings at a random index, i.e., if the string is s, divide it to x and y where s = x + y.
Randomly decide to swap the two substrings or to keep them in the same order. i.e., after this step, s may become s = x + y or s = y + x.
Apply step 1 recursively on each of the two substrings x and y.
Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false."""


class ThinkingBudgetClient:
   def __init__(self, base_url: str, api_key: str, tokenizer_name_or_path: str):
       self.base_url = base_url
       self.api_key = api_key
       self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
       self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)


   def chat_completion(
       self,
       model: str,
       messages: List[Dict[str, Any]],
       max_thinking_budget: int = 512,
       max_tokens: int = 1024,
       **kwargs,
   ) -> Dict[str, Any]:
       assert (
           max_tokens > max_thinking_budget
       ), f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {max_thinking_budget=}"

       # 1. first call chat completion to get reasoning content
       response = self.client.chat.completions.create(
           model=model, 
           messages=messages, 
           max_tokens=max_thinking_budget, 
           logprobs=True,
           top_logprobs=16,
           **kwargs
       )
       content = response.choices[0].message.content
       thinking_logprobs = response.choices[0].logprobs

       reasoning_content = content
       if not "</think>" in reasoning_content:
           # reasoning content is too long, closed with a period (.)
           reasoning_content = f"{reasoning_content}.\n</think>\n\n"
       reasoning_tokens_len = len(
           self.tokenizer.encode(reasoning_content, add_special_tokens=False)
       )
       remaining_tokens = max_tokens - reasoning_tokens_len
       assert (
           remaining_tokens > 0
       ), f"remaining tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the max_thinking_budget."

       # 2. append reasoning content to messages and call completion
       messages.append({"role": "assistant", "content": reasoning_content})
       prompt = self.tokenizer.apply_chat_template(
           messages,
           tokenize=False,
           continue_final_message=True, # not add special token to the end
       )
       
       response = self.client.completions.create( # use completion
           model=model, 
           prompt=prompt, 
           max_tokens=max_tokens, 
           logprobs=16,
           echo=False,
           **kwargs
       )
       final_logprobs = response.choices[0].logprobs

       response_data = {
           "reasoning_content": reasoning_content.strip().strip("</think>").strip(),
           "content": response.choices[0].text,
           "finish_reason": response.choices[0].finish_reason,
           'thinking_logprobs': thinking_logprobs,
           'final_logprobs': final_logprobs,
       }
       return response_data

tokenizer_name_or_path = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
client = ThinkingBudgetClient(
   base_url="http://localhost:8000/v1",  # Nano 9B v2 deployed in thinking mode
   api_key="EMPTY",
   tokenizer_name_or_path=tokenizer_name_or_path,
)

messages = [
       {"role": "system", "content": "You are a helpful assistant. /think"},
       {"role": "user", "content": prompt},
]

result = client.chat_completion(
   model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
   messages=messages,
   max_thinking_budget=budget,
   max_tokens=max_tokens,
   temperature=0.6,
   top_p=0.95,
)

thinking_logprobs = result['thinking_logprobs']
final_logprobs = result['final_logprobs']

entropies = []
for i, token_data in enumerate(thinking_logprobs.content):
    token_logprobs = [lp.logprob for lp in token_data.top_logprobs]
    probs = np.exp(token_logprobs)
    entropy = -np.sum(probs * token_logprobs)
    entropies.append(entropy)
final_entropy_idx = len(thinking_logprobs.content)
for token_logprobs in final_logprobs.top_logprobs:
    probs = np.exp(list(token_logprobs.values()))
    logprobs = list(token_logprobs.values())
    entropy = -np.sum(probs * logprobs)
    entropies.append(entropy)

k = 8
# find largest k entropies and indices
top_k_entropies, top_k_indices = zip(*heapq.nlargest(k, zip(entropies, range(len(entropies)))))
for entropy, idx in zip(top_k_entropies, top_k_indices):
    print(f"Token idx: {idx}, entropy: {entropy}")

window_size = 8
smoothed = np.convolve(entropies, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(16, 6))
plt.plot(entropies, color='lightblue', alpha=0.3, linewidth=1, label='Token Entropy')
plt.plot(range(window_size//2, len(smoothed)+window_size//2), smoothed, color='blue', linewidth=2, label=f'Smoothed (window size {window_size})')

threshold = np.percentile(entropies, 90)
plt.axhline(
    y=threshold,
    color='red',
    linestyle='--',
    alpha=0.5,
    label=f'Threshold at 90th percentile: {threshold:.2f}'
)
plt.axvline(
    x=final_entropy_idx,
    color='green',
    linestyle='--',
    alpha=0.7,
    label='End of thinking budget'
)
plt.xlabel('Token Position', fontsize=12)
plt.ylabel('Entropy', fontsize=12)
plt.title('Token-level Entropy Analysis', fontsize=14, fontweight='bold')
plt.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"entropy/token_entropy_leetcode_budget_{budget}.png")