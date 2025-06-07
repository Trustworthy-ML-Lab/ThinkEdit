import gc
import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import json
from utils import get_think_length, extract_answer, model_dict
from math_grader import math_equal, strip_string

np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-qwen-1.5b", choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b", "deepseek-qwen-32b"])
parser.add_argument("--control", type=str, default="thinking_length_mlp", choices=["thinking_length_mlp", "thinking_length_attn"])
parser.add_argument("--direction_weight", type=float, default=0.00)
args = parser.parse_args()


model_path = model_dict[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

THINK_START_TOKEN_ID = tokenizer.encode("<think>", add_special_tokens=False)[0]
THINK_END_TOKEN_ID = tokenizer.encode("</think>", add_special_tokens=False)[0]

dataset = load_dataset("csv", data_files="math_level5_140_test_examples.csv")
math = dataset["train"]  # Usually "train" if there's only one split
if args.control == "thinking_length_mlp":
    direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_mlp.pt").to(device)
elif args.control == "thinking_length_attn":
    direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_attn.pt").to(device)


if "mlp" in args.control:
    def adjust_residual_hook(layer_idx):
        def hook_fn(module, input, output):
            return output + args.direction_weight * direction[layer_idx]
        return hook_fn

    print("add mlp hook")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_hook(adjust_residual_hook(i))
elif "attn" in args.control:
    def adjust_residual_hook(layer_idx):
        def hook_fn(module, input, output):
            return (output[0] + args.direction_weight * direction[layer_idx],) + output[1:]
        return hook_fn

    print("add attn hook")
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(adjust_residual_hook(i))

responses = []
think_lengths = []
correctness = []
correct_count = 0
total_count = 0


for q, a in zip(math['problem'], math['answer']):
    prompt = f"<｜User｜>{q}<｜Assistant｜>"
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=toks['input_ids'],
            attention_mask=toks['attention_mask'],
            max_new_tokens=16384,
            do_sample=True
        )
    
    think_length, complete_thinking = get_think_length(output_ids[0], THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, max_length=16384)
    print(think_length)

    think_lengths.append(think_length)

    if think_length == -1:
        continue

    output_text = tokenizer.decode(output_ids[0])
        
    if think_length >= 16384:
        output_text += "\n</think>\n\nYeah, I think that's right.\n\n**Final Answer**\n"
        toks = tokenizer(output_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=toks['input_ids'],
                attention_mask=toks['attention_mask'],
                max_new_tokens=256,  # Generate only the final answer
                do_sample=True
            )
        output_text = tokenizer.decode(output_ids[0])

    predicted_answer = strip_string(extract_answer(output_text))
        
    print(output_text)

    responses.append(output_text)
    
    print(predicted_answer)

    ground_truth = int(a)
    
    total_count += 1
    if math_equal(predicted_answer, ground_truth):
        correct_count += 1
        print(1)
        correctness.append(1)
    else:
        print(0)
        correctness.append(0)

accuracy = correct_count / total_count if total_count > 0 else 0
print(f"Accuracy after times {args.direction_weight}: {accuracy:.4f}")
print(f"Average thinking length: {sum(think_lengths) / len(think_lengths) if think_lengths else 0}")

results = {
    "responses": responses,
    "think_lengths": think_lengths,
    "avg_thinking_length": sum(think_lengths) / len(think_lengths),
    "correctness": correctness,
    "accuracy": accuracy
}
os.makedirs(f"math_level5_all_layer_thinking_length_steering_results/{args.control}/{args.model}", exist_ok=True)
with open(f"math_level5_all_layer_thinking_length_steering_results/{args.control}/{args.model}/{args.direction_weight}.json", "w") as f:
    json.dump(results, f, indent=4)

# Plot thinking length distribution
plt.figure(figsize=(10, 6))
bin_edges = list(range(0, 16600, 200)) + [17800, 18000]
plt.hist([t if t < 16384 else 18000 for t in think_lengths], bins=bin_edges, alpha=0.7, edgecolor='black', label="Thinking Length")
plt.xticks(list(range(0, 18000, 2000)) + [18000], labels=[str(i) for i in range(0, 18000, 2000)] + ['>16384'])
plt.xlabel("Thinking Length (tokens)")
plt.ylabel("Frequency")
plt.title(f"Distribution of Thinking Length After Steering (times {args.direction_weight})")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"math_level5_all_layer_thinking_length_steering_results/{args.control}/{args.model}/{args.direction_weight}.png")
