import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_dict

# Hugging Face authentication not needed anymore

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-qwen-1.5b", choices=["deepseek-qwen-1.5b", "deepseek-llama3-8b", "deepseek-qwen-14b"])
parser.add_argument("--intervention_weight", type=float, default=1.0, help="Intervention strength")
args = parser.parse_args()

# Load model and tokenizer
model_path = model_dict[args.model]
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Model config
num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads

# Define heads to edit
heads = []
if args.model == "deepseek_llama3":
    heads = [(13, 31), (27, 14), (30, 18), (21, 19), (16, 31), (17, 13), (25, 11), (12, 0), (26, 10), (11, 2),
             (30, 17), (26, 11), (13, 29), (9, 4), (22, 6), (16, 6), (18, 12), (12, 1), (12, 23), (12, 12)]
elif args.model == "deepseek_qwen_1.5b":
    heads = [(21, 2), (17, 7), (16, 9), (21, 3), (17, 11), (15, 6), (21, 1), (25, 3), (19, 9), (24, 3)]
elif args.model == "deepseek_qwen_14b":
    heads = [(36, 9), (36, 7), (28, 21), (31, 1), (42, 11), (28, 32), (45, 9), (34, 0), (28, 24), (27, 21),
             (39, 35), (39, 39), (26, 27), (45, 16), (44, 33), (36, 6), (47, 25), (47, 28), (27, 24), (27, 20),
             (44, 32), (31, 0), (35, 17), (22, 23), (38, 17), (26, 28), (45, 17), (31, 2), (33, 25), (24, 39),
             (39, 23), (31, 15), (24, 37), (28, 22), (45, 8), (36, 8), (30, 30), (42, 14), (30, 32), (38, 14)]

# Function to remove projection along a direction
def remove_projection_along_v(W_o, thinking_direction):
    v_normalized = thinking_direction / torch.norm(thinking_direction)
    projection = torch.outer(torch.matmul(W_o, v_normalized), v_normalized)
    W_o_modified = W_o - args.intervention_weight * projection

    projection_before = torch.norm(torch.matmul(W_o, thinking_direction))
    projection_after = torch.norm(torch.matmul(W_o_modified, thinking_direction))

    print(f"Projection before modification: {projection_before:.4f}")
    print(f"Projection after modification: {projection_after:.4f}")

    return W_o_modified

# Load thinking direction
thinking_direction = torch.load(f"directions/{args.model}_thinking_length_direction_gsm8k_attn.pt").to(device)
thinking_direction = thinking_direction / torch.norm(thinking_direction, dim=-1, keepdim=True)
thinking_direction = -thinking_direction

# Apply intervention
for layer_idx, head_idx in heads:
    start_idx = head_idx * head_dim
    end_idx = (head_idx + 1) * head_dim

    o_proj_weight = model.model.layers[layer_idx].self_attn.o_proj.weight.detach().clone()
    W_o = o_proj_weight[:, start_idx:end_idx].T.float()

    # Modify
    W_o_modified = remove_projection_along_v(W_o, thinking_direction[layer_idx][0].float())

    # Update model
    o_proj_weight[:, start_idx:end_idx] = W_o_modified.T.to(torch.bfloat16)
    model.model.layers[layer_idx].self_attn.o_proj.weight = torch.nn.Parameter(o_proj_weight)

# Save the edited model locally
save_dir = f"ThinkEdit_models/ThinkEdit-{args.model}"
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model saved successfully to: {save_dir}")
