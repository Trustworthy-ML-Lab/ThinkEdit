import os
import json
import re
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoConfig, GenerationConfig
from utils import model_dict

# ----------------------
# Argument parsing
# ----------------------
parser = argparse.ArgumentParser(
    description="Evaluate thinking length of LLM responses on GSM8K using offline vLLM"
)
parser.add_argument(
    "--model", type=str,
    default="deepseek-qwen-1.5b",
    choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b", "deepseek-qwen-32b"],
    help="Model to evaluate"
)
parser.add_argument(
    "--tensor_parallel_size", type=int,
    default=1,
    help="Tensor parallel size for vLLM"
)
args = parser.parse_args()

# Set random seed for reproducibility
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)

# ----------------------
# Load dataset
# ----------------------
gsm8k = load_dataset('openai/gsm8k', 'main', split='train[:2000]')
questions = gsm8k['question']

# ----------------------
# Prepare prompts with model-specific template
# ----------------------
prompts = []
for q in questions:
    prompts.append(f"<｜User｜>{q}<｜Assistant｜>")

# ----------------------
# Initialize vLLM offline LLM
# ----------------------
model_path = model_dict[args.model]
llm = LLM(
    model=model_path,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=4096 + 2048
)
# Tokenizer for token counting
tokenizer = llm.get_tokenizer()

# ----------------------
# Load only config to retrieve generation defaults
# ----------------------
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# Convert to GenerationConfig if needed
gen_cfg = None
try:
    gen_cfg = GenerationConfig.from_pretrained(model_path)
except Exception:
    # Fallback: build from model config attributes
    gen_cfg = GenerationConfig(**{k: getattr(config, k) for k in ['temperature', 'top_k', 'top_p', 'repetition_penalty'] if hasattr(config, k)})

# ----------------------
# Build sampling parameters from model's generation_config
# ----------------------
sampling_params = SamplingParams(
    temperature=getattr(gen_cfg, 'temperature', 0.6),
    top_p=getattr(gen_cfg, 'top_p', 0.95),
    top_k=getattr(gen_cfg, 'top_k', None),
    repetition_penalty=getattr(gen_cfg, 'repetition_penalty', 1.0),
    max_tokens=4096
)

# ----------------------
# Helper to extract thinking section
# ----------------------
def extract_thinking(response_text):
    match = re.search(r"(<think>.*?</think>)", response_text, re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        length = len(tokenizer(thinking, return_tensors='np')['input_ids'][0])
        return thinking, int(length)
    return "", -1

# ----------------------
# Run offline batch inference
# ----------------------
print(f"Running offline batch inference on {len(prompts)} examples with model {args.model}...")
outputs = llm.generate(prompts, sampling_params)
print(outputs)
# ----------------------
# Process outputs
# ----------------------
responses_data = []
thinking_lengths = []
for question, batch_result in zip(questions, outputs):
    text = batch_result.outputs[0].text.strip()
    thinking, length = extract_thinking(text)
    responses_data.append({
        "question": question,
        "response": text,
        "thinking": thinking,
        "thinking_length": length
    })
    thinking_lengths.append(length)

# ----------------------
# Save results using original filenames
# ----------------------
os.makedirs("responses", exist_ok=True)
json_path = f"responses/{args.model}_gsm8k.json"
with open(json_path, 'w') as f:
    json.dump(responses_data, f, indent=4)
print(f"Saved JSON results to {json_path}")

# ----------------------
# Plot thinking length distribution
# ----------------------
plt.figure(figsize=(10, 6))
plt.hist(thinking_lengths, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Thinking Length (tokens)")
plt.ylabel("Frequency")
plt.title("Distribution of Thinking Length in Model Responses")
plt.grid(axis='y', linestyle='--', alpha=0.7)
png_path = f"responses/{args.model}_thinking_length_distribution_gsm8k.png"
plt.savefig(png_path)
print(f"Saved histogram plot to {png_path}")
