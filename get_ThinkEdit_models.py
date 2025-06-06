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
parser.add_argument("--model", type=str, default="deepseek-qwen-1.5b")
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
if args.model == "deepseek-qwen-1.5b":
    heads = [(21, 2), (17, 7), (16, 9), (21, 3), (25, 3), (17, 11), (15, 6), (21, 1), (19, 9), (15, 11), (24, 3), (19, 11), (17, 9), (19, 10), (20, 8), (24, 4), (11, 6), (22, 3), (22, 4), (23, 10)]
elif args.model == "deepseek-llama3-8b":
    heads = [(13, 31), (27, 14), (17, 13), (21, 19), (30, 18), (16, 31), (12, 0), (25, 11), (26, 10), (11, 2), (30, 17), (26, 11), (13, 29), (9, 4), (12, 1), (16, 6), (22, 6), (12, 23), (18, 12), (6, 25), (16, 7), (16, 29), (15, 5), (12, 12), (11, 18), (26, 8), (24, 13), (7, 21), (31, 26), (15, 19), (28, 20), (19, 18), (8, 21), (16, 13), (30, 6), (27, 12), (22, 5), (17, 14), (31, 0), (10, 26)]
elif args.model == "deepseek-qwen-14b":
    heads = [(36, 9), (36, 7), (28, 21), (31, 1), (28, 32), (42, 11), (45, 9), (39, 39), (28, 24), (34, 0), (39, 35), (27, 21), (26, 27), (45, 16), (36, 6), (47, 28), (47, 25), (27, 20), (44, 33), (27, 24), (31, 0), (35, 17), (38, 17), (22, 23), (31, 2), (45, 17), (33, 25), (44, 32), (26, 28), (24, 39), (31, 15), (45, 8), (39, 23), (30, 30), (36, 8), (28, 22), (30, 12), (30, 32), (28, 31), (42, 14), (23, 24), (24, 37), (38, 14), (35, 20), (33, 5), (40, 16), (30, 19), (33, 9), (33, 6), (44, 4), (47, 35), (33, 28), (40, 34), (40, 30), (26, 37), (35, 22), (37, 22), (30, 10), (30, 6), (37, 23), (30, 14), (46, 36), (39, 36), (33, 7), (20, 34), (7, 36), (45, 6), (30, 17), (38, 11), (38, 10), (42, 32), (38, 15), (47, 26), (35, 18), (34, 4), (19, 14), (30, 34), (22, 15), (32, 13), (33, 8)]
elif args.model == "deepseek-qwen-32b":
    heads = [(44, 21), (52, 7), (61, 9), (58, 30), (52, 9), (42, 27), (58, 11), (55, 39), (38, 23), (43, 21), (47, 1), (55, 35), (44, 32), (61, 8), (44, 24), (28, 21), (61, 16), (51, 17), (50, 0), (43, 24), (39, 24), (61, 17), (47, 15), (52, 6), (31, 1), (44, 22), (58, 32), (47, 0), (28, 24), (54, 17), (40, 39), (55, 37), (58, 14), (56, 34), (46, 12), (51, 18), (42, 28), (47, 2), (42, 25), (42, 37), (56, 30), (55, 36), (48, 12), (46, 32), (60, 33), (63, 30), (61, 7), (44, 31), (43, 20), (58, 29), (40, 37), (38, 15), (61, 6), (33, 0), (53, 22), (39, 23), (60, 32), (35, 14), (54, 15), (49, 5), (49, 28), (46, 19), (39, 5), (39, 35), (49, 7), (56, 16), (54, 14), (7, 36), (63, 33), (33, 39), (51, 15), (50, 4), (36, 33), (57, 1), (27, 24), (42, 26), (49, 9), (46, 10), (55, 23), (48, 13), (53, 23), (46, 14), (62, 36), (46, 17), (57, 4), (38, 20), (35, 6), (35, 2), (40, 36), (48, 25), (32, 38), (35, 4), (60, 4), (46, 34), (46, 18), (26, 27), (46, 6), (51, 22), (51, 20), (36, 32), (51, 6), (37, 17), (51, 35), (45, 22), (49, 25), (54, 18), (47, 34), (53, 37), (45, 32), (35, 12), (42, 38), (53, 24), (43, 13), (38, 6), (39, 6), (41, 22), (23, 24), (41, 8), (57, 31), (45, 20), (40, 38), (22, 15), (43, 9), (56, 20), (57, 0), (44, 25), (49, 8), (60, 0), (43, 25), (59, 22), (60, 16), (41, 20), (46, 31), (60, 17), (47, 30), (32, 35), (48, 14), (55, 7), (39, 21), (56, 17), (56, 33), (54, 11), (39, 8), (40, 1), (44, 36), (16, 38), (57, 2), (26, 28), (49, 23), (27, 21), (54, 10), (30, 34), (62, 35), (57, 35), (55, 22), (48, 26), (53, 5), (36, 7), (42, 3), (46, 33)] 

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
