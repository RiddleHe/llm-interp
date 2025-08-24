from transformers import AutoModelForCausalLM, AutoTokenizer, TopPLogitsWarper, TemperatureLogitsWarper
import os
import re
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import json

parser = argparse.ArgumentParser()
parser.add_argument("--token_budget", type=int, default=1024)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--temperature", type=float, default=1.1)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--prompt_type", type=str, default="livecodebench")
parser.add_argument("--swap_token", action="store_true", default=False)
args = parser.parse_args()

token_budget = args.token_budget
max_tokens = args.max_tokens
temperature = args.temperature
top_p = args.top_p
prompt_type = args.prompt_type

with open(f"prompts.json", "r") as f:
    prompts = json.load(f)
prompt = prompts[prompt_type][0]

model_name_or_path = "ByteDance-Seed/Seed-OSS-36B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here
messages = [
    {"role": "user", "content": prompt},
]
tokenized_messages = tokenizer.apply_chat_template(
  messages, 
  tokenize=False,
  add_generation_prompt=True, 
  thinking_budget=token_budget # control the thinking budget
)
print(f"Tokenized messages: {tokenized_messages}")

generated_ids = tokenizer(tokenized_messages, return_tensors="pt").input_ids.to(model.device)
prompt_len = generated_ids.shape[1]
past_key_values = None

temp_warper = TemperatureLogitsWarper(temperature)
top_p_warper = TopPLogitsWarper(top_p)

budget_reflection_start = -1
budget_reflection_stop = -1
perform_swap = False

for step in tqdm(range(max_tokens), desc="Generating tokens"):
    with torch.no_grad():
        output = model(
            generated_ids if step == 0 else generated_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
    logits = output.logits[:, -1, :]
    past_key_values = output.past_key_values

    logits = temp_warper(generated_ids, logits)
    logits = top_p_warper(generated_ids, logits)

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    if next_token.item() == 5: # <seed:cot_budget_reflect>
        budget_reflection_start = step
        print(f"COT budget reflection starts at step {step} with total tokens {step + prompt_len + 1}")
        if step > token_budget:
            print("COT budget reflection exceeds the budget")
        if args.swap_token:
            perform_swap = not perform_swap
            if perform_swap:
                next_token = tokenizer.encode(" ")
                next_token = torch.tensor(next_token).unsqueeze(0).to(generated_ids.device)
                print(f"Swapped token to empty space token")

    elif next_token.item() == 6: # </seed:cot_budget_reflect>
        budget_reflection_stop = step
        print(f"COT budget reflection stops at step {step} with total tokens {step + prompt_len + 1}")
        if budget_reflection_start != -1:
            reflection_text = tokenizer.decode(generated_ids[0, prompt_len + budget_reflection_start :])
            print(f"COT budget reflection text: {reflection_text}")
        budget_reflection_start = -1
        budget_reflection_stop = -1

    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    if next_token.item() == tokenizer.eos_token_id:
        break

output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
print(output_text)
