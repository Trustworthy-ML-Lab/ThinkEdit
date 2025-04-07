import gc
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import re
from utils import model_dict

from transformers.utils import logging
logging.set_verbosity_error()

import pickle
import time

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-qwen-1.5b", choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b"])
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

gsm8k = load_dataset('openai/gsm8k', 'main', split='train[:2000]')


model_path = model_dict[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token

def extract_thinking(response):
    """Extracts the thinking part from response text, including the <think> tags."""
    match = re.search(r"(<think>.*?</think>)", response, re.DOTALL)
    if match:
        thinking_text = match.group(1).strip()
        return thinking_text, len(tokenizer(thinking_text, return_tensors='pt')['input_ids'][0])
    return "", -1

responses_data = []
thinking_lengths = []

total = len(gsm8k['question'])
for idx, q in enumerate(gsm8k['question']):
    print(f"Processing {idx+1}/{total}...", end='\r')
    toks = tokenizer(f"<｜User｜>{q}<｜Assistant｜>", return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device), max_new_tokens=4096)[0]                        
    output = tokenizer.decode(output_ids[len(toks['input_ids'][0]):])
    print(output)
    output = output.replace("<\uff5cend\u2581of\u2581sentence\uff5c>", "")
    thinking_part, thinking_length = extract_thinking(output)
    thinking_lengths.append(thinking_length)

    # print(output)
    # print(thinking_length)
    
    responses_data.append({
        "question": q,
        "response": output,
        "thinking": thinking_part,
        "thinking_length": thinking_length
    })

os.makedirs("responses", exist_ok=True)
with open(f"responses/{args.model}_gsm8k.json", 'w') as f:
    json.dump(responses_data, f, indent=4)

# Plot thinking length distribution
plt.figure(figsize=(10, 6))
plt.hist(thinking_lengths, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel("Thinking Length (tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Thinking Length in Model Responses")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"responses/{args.model}_thinking_length_distribution_gsm8k.png")
