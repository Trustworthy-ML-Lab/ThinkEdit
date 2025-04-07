import gc
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utils import model_dict

from transformers.utils import logging
logging.set_verbosity_error()

import pickle
import time
import json
import math
import re

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek_llama3")
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Load JSON file with response data
json_file_path = f"responses/{args.model}_gsm8k.json"
with open(json_file_path, 'r') as f:
    responses_data = json.load(f)

# Filter examples based on thinking length
valid_responses = [ex for ex in responses_data if ex['thinking_length'] != -1]
long_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] > 1000]
short_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] < 100]
print(len(long_thinking_examples))
print(len(short_thinking_examples))

model_path = model_dict[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token

residual_outputs = []

def capture_residual_hook():
    def hook_fn(module, input, output):
        residual_outputs.append(input[0].detach())
    return hook_fn

for layer in model.model.layers:
    layer.post_attention_layernorm.register_forward_hook(capture_residual_hook())

long_thinking_embeddings = []
short_thinking_embeddings = []

for example in long_thinking_examples:
    toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>").input_ids
    start = len(toks)
    toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>{example['thinking']}").input_ids
    end = len(toks)
    toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>{example['thinking']}", return_tensors="pt")
    with torch.no_grad():
        _ = model(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device))
    long_thinking_embeddings.append(torch.stack(residual_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu())
    residual_outputs = []

for example in short_thinking_examples:
    toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>").input_ids
    start = len(toks)
    toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>{example['thinking']}").input_ids
    end = len(toks)
    toks = tokenizer(f"<｜User｜>{example['question']}<｜Assistant｜>{example['thinking']}", return_tensors="pt")
    with torch.no_grad():
        _ = model(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device))
    short_thinking_embeddings.append(torch.stack(residual_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu())
    residual_outputs = []


mean_embedding_long = torch.stack(long_thinking_embeddings, dim=0).mean(dim=0)
mean_embedding_short = torch.stack(short_thinking_embeddings, dim=0).mean(dim=0)
thinking_length_direction = mean_embedding_long - mean_embedding_short
os.makedirs("directions", exist_ok=True)
torch.save(thinking_length_direction, f"directions/{args.model}_thinking_length_direction_gsm8k_attn.pt")
