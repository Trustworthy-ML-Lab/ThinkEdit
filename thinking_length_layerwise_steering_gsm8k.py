import gc
import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
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
parser.add_argument("--layer", type=int, default=0)
args = parser.parse_args()


model_path = model_dict[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

THINK_START_TOKEN_ID = tokenizer.encode("<think>", add_special_tokens=False)[0]
THINK_END_TOKEN_ID = tokenizer.encode("</think>", add_special_tokens=False)[0]

gsm8k = load_dataset('openai/gsm8k', 'main', split='test[:100]')
if args.control == "thinking_length_mlp":
    direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_mlp.pt").to(device)
elif args.control == "thinking_length_attn":
    direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_attn.pt").to(device)

if "mlp" in args.control:
    print("add mlp hook")
    def adjust_residual_hook(layer_idx):
        def hook_fn(module, input, output):
            return output + args.direction_weight * direction[layer_idx]
        return hook_fn

    model.model.layers[args.layer].mlp.register_forward_hook(adjust_residual_hook(args.layer))
elif "attn" in args.control:
    def adjust_residual_hook(layer_idx):
        def hook_fn(module, input, output):
            return (output[0] + args.direction_weight * direction[layer_idx],) + output[1:]
        return hook_fn

    print("add attn hook")
    model.model.layers[args.layer].self_attn.register_forward_hook(adjust_residual_hook(args.layer))

responses = []
think_lengths = []
correctness = []
correct_count = 0
total_count = 0

def extract_ground_truth(answer_text):
    match = re.search(r'####\s*(\d+)', answer_text)
    return int(match.group(1)) if match else None

for q, a in zip(gsm8k['question'], gsm8k['answer']):
    prompt = f"<｜User｜>{q}<｜Assistant｜>"
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=toks['input_ids'],
            attention_mask=toks['attention_mask'],
            max_new_tokens=8192,
            do_sample=True
        )
    
    think_length, complete_thinking = get_think_length(output_ids[0], THINK_START_TOKEN_ID, THINK_END_TOKEN_ID, max_length=8192)
    print(think_length)

    think_lengths.append(think_length)

    if think_length == -1:
        continue

    output_text = tokenizer.decode(output_ids[0])
        
    if think_length >= 8192:
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

    ground_truth = extract_ground_truth(a)
    
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
os.makedirs(f"gsm8k_layerwise_thinking_length_steering_results/{args.control}/{args.model}", exist_ok=True)
with open(f"gsm8k_layerwise_thinking_length_steering_results/{args.control}/{args.model}/{args.layer}_{args.direction_weight}.json", "w") as f:
    json.dump(results, f, indent=4)