import gc
import os
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from utils import model_dict

from transformers.utils import logging
logging.set_verbosity_error()

import json

# Reproducibility
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="deepseek-qwen-1.5b"
)
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Prompt templates (minimal change)
TEMPLATE_BASE = "<｜User｜>{question}<｜Assistant｜>"
TEMPLATE_FULL = "<｜User｜>{question}<｜Assistant｜>{thinking}"

# Load JSON file with response data
json_file_path = f"responses/{args.model}_gsm8k.json"
with open(json_file_path, 'r') as f:
    responses_data = json.load(f)

# Filter examples based on thinking length
long_thinking_threshold = 1000
short_thinking_threshold = 100

valid_responses = [ex for ex in responses_data if ex['thinking_length'] != -1]
long_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] > long_thinking_threshold]
short_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] < short_thinking_threshold]
print(len(long_thinking_examples))
print(len(short_thinking_examples))

model_path = model_dict[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token

long_thinking_embeddings = []
short_thinking_embeddings = []

# Process long thinking examples
for example in long_thinking_examples:
    base_str = TEMPLATE_BASE.format(question=example['question'])
    start = len(tokenizer(base_str).input_ids)
    full_str = TEMPLATE_FULL.format(question=example['question'], thinking=example['thinking'])
    end = len(tokenizer(full_str).input_ids)
    toks = tokenizer(full_str, return_tensors="pt")
    with torch.no_grad():
        residual_outputs = model(
            input_ids=toks['input_ids'].to(device),
            attention_mask=toks['attention_mask'].to(device),
            output_hidden_states=True
        ).hidden_states[1:]
    long_thinking_embeddings.append(
        torch.stack(residual_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu()
    )

# Process short thinking examples
for example in short_thinking_examples:
    base_str = TEMPLATE_BASE.format(question=example['question'])
    start = len(tokenizer(base_str).input_ids)
    full_str = TEMPLATE_FULL.format(question=example['question'], thinking=example['thinking'])
    end = len(tokenizer(full_str).input_ids)
    toks = tokenizer(full_str, return_tensors="pt")
    with torch.no_grad():
        residual_outputs = model(
            input_ids=toks['input_ids'].to(device),
            attention_mask=toks['attention_mask'].to(device),
            output_hidden_states=True
        ).hidden_states[1:]
    short_thinking_embeddings.append(
        torch.stack(residual_outputs, dim=0)[:, :, start-1:end-1, :].mean(dim=2).cpu()
    )

mean_embedding_long = torch.stack(long_thinking_embeddings, dim=0).mean(dim=0)
mean_embedding_short = torch.stack(short_thinking_embeddings, dim=0).mean(dim=0)
thinking_length_direction = mean_embedding_long - mean_embedding_short
os.makedirs("directions", exist_ok=True)
torch.save(thinking_length_direction, f"directions/{args.model}_thinking_length_direction_gsm8k_mlp.pt")
