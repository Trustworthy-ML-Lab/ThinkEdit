import gc
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pickle
import time
import json
import math
import re
import matplotlib.pyplot as plt
from utils import model_dict

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str,
    default="deepseek-qwen-1.5b",
    choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b", "deepseek-qwen-32b", "qwq-32b", "s1-32b"]
)
parser.add_argument("--layer_start", type=int, default=0, help="Start layer for visualization")
parser.add_argument("--layer_end", type=int, default=-1, help="End layer for visualization")
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Prompt templates
PROMPT_BASE = "<｜User｜>{q}<｜Assistant｜>"
PROMPT_FULL = "<｜User｜>{q}<｜Assistant｜>{thinking}"

def top_k_head(matrix, k=20, r=True):
    flattened = [(value, (i, j)) for i, row in enumerate(matrix) for j, value in enumerate(row)]
    return sorted(flattened, key=lambda x: x[0], reverse=r)[:k]

json_file_path = f"responses/{args.model}_gsm8k.json"
with open(json_file_path, 'r') as f:
    responses_data = json.load(f)


short_thinking_threshold = 100
valid_responses = [ex for ex in responses_data if ex['thinking_length'] != -1]
short_thinking_examples = [ex for ex in valid_responses if ex['thinking_length'] < short_thinking_threshold]
print(len(short_thinking_examples))

model_path = model_dict[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
model.generation_config.do_sample = True
tokenizer.pad_token = tokenizer.eos_token

thinking_length_direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_attn.pt").to(device)
thinking_length_direction = thinking_length_direction / torch.norm(thinking_length_direction, dim=-1, keepdim=True)

num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads
attn_contribution = []

def capture_attn_contribution_hook():
    def hook_fn(module, input, output):
        attn_out = input[0].detach()[0, :, :]
        attn_out = attn_out.reshape(attn_out.size(0), num_heads, head_dim)
        o_proj = module.weight.detach().clone()
        o_proj = o_proj.reshape(hidden_size, num_heads, head_dim).permute(1, 2, 0).contiguous()
        attn_contribution.append(torch.einsum("snk,nkh->snh", attn_out, o_proj))
    return hook_fn

for layer in model.model.layers:
    layer.self_attn.o_proj.register_forward_hook(capture_attn_contribution_hook())

avg_contribution = np.zeros((num_layers, num_heads))
for i, example in enumerate(short_thinking_examples):
    print(f"process example {i}", end='\r')
    toks = tokenizer(PROMPT_BASE.format(q=example['question'])).input_ids
    start = len(toks)
    toks = tokenizer(PROMPT_FULL.format(q=example['question'], thinking=example['thinking'])).input_ids
    end = len(toks)
    toks = tokenizer(PROMPT_FULL.format(q=example['question'], thinking=example['thinking']), return_tensors="pt")
    with torch.no_grad():
        _ = model(input_ids=toks['input_ids'].to(device), attention_mask=toks['attention_mask'].to(device))
        attn_mean_contributions = [tensor[start-1:end-1, :, :].mean(dim=0) for tensor in attn_contribution]
        all_head_contributions = torch.stack(attn_mean_contributions, dim=0)
        dot_products = torch.einsum('ijl,il->ij', all_head_contributions.float(), -thinking_length_direction[:, 0].float())
        avg_contribution += dot_products.cpu().numpy()
    attn_contribution = []

avg_contribution = avg_contribution / len(short_thinking_examples)
layer_start = max(0, args.layer_start)
layer_end = num_layers if args.layer_end == -1 else min(num_layers, args.layer_end)
if "1.5b" in args.model.lower():
    k = 20
elif "8b" in args.model.lower():
    k = 40
elif "14b" in args.model.lower():
    k = 80
elif "32b" in args.model.lower():
    k = 160

top_k_contributions = top_k_head(avg_contribution[layer_start:layer_end, :], k=k)
print([(c[0], (c[1][0] + layer_start, c[1][1])) for c in top_k_contributions])
print(f"top {k} short thinking heads:", [(c[1][0] + layer_start, c[1][1]) for c in top_k_contributions])

max_abs_value = np.abs(avg_contribution[layer_start:layer_end, :]).max()

plt.figure(figsize=(12, 10))
plt.imshow(avg_contribution[layer_start:layer_end, :], cmap='coolwarm', aspect='auto', vmin=-max_abs_value, vmax=max_abs_value)
plt.colorbar(label='Average short thinking contribution')
plt.title(f'Heatmap of Average short thinking contribution (Layers {layer_start}-{layer_end})')
plt.xlabel('Heads')
plt.ylabel('Layers')
plt.xticks(ticks=np.arange(num_heads), labels=[f'H{i}' for i in range(num_heads)], fontsize=6, rotation=45)
plt.yticks(ticks=np.arange(layer_end - layer_start), labels=[f'L{i+layer_start}' for i in range(layer_end - layer_start)], fontsize=8)
plt.tight_layout()
plt.savefig(f'{args.model}_short_thinking_attn_head_heatmap_{layer_start}_{layer_end}.png', dpi=300)
plt.close()
