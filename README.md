# ThinkEdit

This is the official repository for the paper: [**ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models**](https://arxiv.org/abs/2503.22048). Here is our 5 min read [project website](https://lilywenglab.github.io/ThinkEdit/).

## 🆕 Updates

- **2025/08/20** — 🎉 Our paper has been **accepted to EMNLP 2025 main conference**!
- **2025/06/06** — 🚀 Released improved **ThinkEdit models** with **4% short reasoning heads edited** and introduced the larger **ThinkEdit-deepseek-qwen-32b** model. Please see our updated paper.



## Overview

<p align="center">
  <img src="./fig/overview.png" width="80%" />
</p>

## Set Up

```bash
pip install -r requirements.txt
```

If you want to skip all the steps and directly access the resulting output files, you can download them through:

```bash
gdown https://drive.google.com/uc?id=135ttUxdNc4gQkmbZlpr3ecbhA3_392PE
```
and then unzip the file
```bash
unzip ThinkEdit.zip
```

## Steer along Reasoning Length Direction

### Step 1: Generate responses for probing from GSM8k

First, collect the responses from the reasoning models and store them in `responses/` for extracting hidden states later:

```bash
python generate response_gsm8k.py
```

Specify the `--model` argument: `deepseek-qwen-1.5b`, `deepseek-llama3-8b`, `deepseek-qwen-14b`, `deepseek-qwen-32b`.

### Step 2: Extract the Reasoning Length Direction

Next, extract the layerwise directions from Self-Attn or MLP and store them in `directions/`:

```bash
python extract_thinking_length_directiongsm8k_attn.py
python extract_thinking_length_directiongsm8k_mlp.py
```

Specify the `--model` argument: `deepseek-qwen-1.5b`, `deepseek-llama3-8b`, `deepseek-qwen-14b`, `deepseek-qwen-32b`.

### Step 3: Steer the reasoning length of the models

Finally, steer the models with the directions and observe changes in accuracy and reasoning length. To evaluate on 200 test examples from gsm8k and store the results in `gsm8k_all_layer_thinking_length_steering_results/`:

```bash
python thinking_length_steering_gsm8k.py
```

Specify the `--model` argument: `deepseek-qwen-1.5b`, `deepseek-llama3-8b`, `deepseek-qwen-14b`, `deepseek-qwen-32b`.

`--control` argument options: `thinking_length_attn`, `thinking_length_mlp`.

The steering strength alpha (`--direction_weight`): we use `-0.08 -0.07 ... 0.07 0.08` in our paper.

Similarly, to evaluate 140 Level-5 examples from MATH and store the results in `math_level5_all_layer_thinking_length_steering_results/`:

```bash
python thinking_length_steering_math_level6.py
```

Specify arguments accordingly.

To steer only one layer each time and store the results in `gsm8k_layerwise_thinking_length_steering_results/`:

```bash
python thinking_length_layerwise_steering_gsm8k.py
```

Specify arguments accordingly. Use `--layer` to specify the layer and set `--direction_weight` to `-1` or `1` (as in our paper). Running the layerwise analysis can take considerable time. We suggest using `automate_layerwise_steering_jobs.sh` to handle the jobs; please modify the script based on your hardware.

## ThinkEdit models: Weight editing short reasoning heads

### Step 1: Find the short reasoning heads

First, identify the short reasoning heads by calculating their per-head contribution to the short reasoning direction:

```bash
python find_short_thinking_attn_heads.py
```

Specify the `--model` argument: `deepseek-qwen-1.5b`, `deepseek-llama3-8b`, `deepseek-qwen-14b`, `deepseek-qwen-32b`.

This will output a list of short reasoning heads and a heatmap figure of every head's contribution.

### Step 2: Perform Weight Editing

Next, perform weight editing to the `o_proj` layer of the short reasoning heads and store the model under `ThinkEdit_models/`:

```bash
python get_ThinkEdit_models.py
```

Specify the `--model` argument: `deepseek-qwen-1.5b`, `deepseek-llama3-8b`, `deepseek-qwen-14b`, `deepseek-qwen-32b`.

We have provided ThinkEdit models on the Huggingface repo:
- `cesun/ThinkEdit-deepseek-qwen-32b`
- `cesun/ThinkEdit-deepseek-qwen-14b`
- `cesun/ThinkEdit-deepseek-llama3-8b`
- `cesun/ThinkEdit-deepseek-qwen-1.5b`

You can skip this step and our evaluation script will directly download the models from Huggingface.

### Step 3: Evaluate the performance of the ThinkEdit models

Finally, evaluate the performance of the original and ThinkEdit models and store the results under `ThinkEdit_model_evaluation_results/`. We use vllm to speed up evaluation:

```bash
CUDA_VISIBLE_DEVICES={your available gpus} python evaluate_ThinkEdit_models.py
```

Specify the `--model` argument: `deepseek-qwen-32b`, `deepseek-qwen-14b`, `deepseek-llama3-8b`, `deepseek-qwen-1.5b`, `ThinkEdit-deepseek-qwen-32b`, `ThinkEdit-deepseek-qwen-14b`, `ThinkEdit-deepseek-llama3-8b`, `ThinkEdit-deepseek-qwen-1.5b`.

`--dataset` argument: `gsm8k`, `mmlu_elementary_math`, `MATH-500`, `MATH-level1`, `MATH-level5`.

`--n_samples` argument: we set this to 10 in our paper, meaning each question is evaluated 10 times.

`--tensor_parallel_size` argument: set this according to your number of GPUs; it should be a factor of the number of attention heads in each model. We recommend setting it to 4.

After you have all the results, run:

```bash
python analyze_ThinkEdit_performance.py
```

to generate the plots and tables presented in our paper. You can expect output similar to the results shown below.

### Results

#### 1. Overall Accuracy


| Model                            | GSM8K               | MMLU Elem. Math | MATH-Lv1         | MATH-Lv5         | MATH500            |
|---------------------------------|---------------------|----------------------|---------------------|---------------------|---------------------|
| ds-qwen-32b               | 92.97 ± 0.39        | 95.93 ± 0.83         | **96.41 ± 0.45**    | 91.27 ± 0.53        | **91.62 ± 0.58**    |
| **ThinkEdit-ds-qwen-32b** | **95.25 ± 0.25**    | **98.02 ± 0.31**     | 96.02 ± 0.42        | **91.31 ± 0.50**    | 91.60 ± 0.65        |
| ds-qwen-14b               | 90.80 ± 0.36        | 95.08 ± 0.65         | 96.32 ± 0.35        | 90.25 ± 0.72        | 91.48 ± 0.55        |
| **ThinkEdit-ds-qwen-14b** | **93.78 ± 0.50**    | **96.56 ± 0.84**     | **96.38 ± 0.52**    | **91.03 ± 0.44**    | **91.92 ± 0.63**    |
| ds-llama3-8b              | 82.26 ± 0.91        | 96.01 ± 0.62         | 93.46 ± 0.84        | 85.49 ± 0.83        | 87.26 ± 1.16        |
| **ThinkEdit-ds-llama3-8b**| **89.44 ± 0.55**    | **96.19 ± 0.73**     | **94.44 ± 0.31**    | **86.49 ± 0.54**    | **88.06 ± 1.09**    |
| ds-qwen-1.5b              | 79.15 ± 1.08        | 68.52 ± 1.56         | 93.00 ± 0.33        | **75.48 ± 0.90**    | 82.22 ± 1.29        |
| **ThinkEdit-ds-qwen-1.5b**| **84.56 ± 0.79**    | **90.66 ± 0.97**     | **93.66 ± 0.62**    | 75.05 ± 0.82        | **82.24 ± 0.89**    |


#### 2. Accuracy on Short Reasoning Cases (Top 5% / 10% / 20%)


| Model                            | GSM8K                            | MMLU Elem. Math              | MATH-Lv1                      | MATH-Lv5                      | MATH500                         |
|---------------------------------|---------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|
| ds-qwen-32b               | 98.31 / 97.18 / 96.20           | 97.78 / 97.03 / 95.87            | 100.00 / 100.00 / **98.97**      | 93.03 / 96.36 / 97.35            | 86.40 / 92.00 / 94.00            |
| **ThinkEdit-ds-qwen-32b** | **98.92** / **97.71** / **97.83** | **97.78** / **97.57** / **97.20** | **100.00** / **100.00** / 98.74   | **98.03** / **98.64** / **97.99** | **92.00** / **94.40** / **95.80** |
| ds-qwen-14b               | **96.31** / 95.65 / 92.93       | 93.89 / **96.22** / 95.60         | 99.52 / **99.30** / 97.70         | 89.39 / 94.32 / 96.25            | 86.40 / 91.40 / 93.50            |
| **ThinkEdit-ds-qwen-14b** | **96.31** / **96.18** / **96.77** | **97.78** / 95.14 / **96.53**     | **99.53** / 98.62 / **98.67**     | **96.67** / **97.88** / **98.11** | **91.20** / **93.20** / **95.00** |
| ds-llama3-8b              | 88.92 / 87.18 / 85.82           | 97.22 / 96.49 / 96.80            | 97.14 / 94.88 / 94.83            | 78.64 / 88.79 / 93.41            | 82.00 / 81.40 / 88.30            |
| **ThinkEdit-ds-llama3-8b**| **97.08** / **95.27** / **93.95** | **97.78** / **98.65** / **97.87** | **100.00** / **99.30** / **98.62** | **95.61** / **96.89** / **97.12** | **92.80** / **93.60** / **94.40** |
| ds-qwen-1.5b              | 88.46 / 87.48 / 85.02           | 62.78 / 62.16 / 60.53            | **97.62** / 95.12 / 93.91         | 91.52 / 95.00 / 95.72            | 82.40 / 89.80 / 93.40            |
| **ThinkEdit-ds-qwen-1.5b**| **92.62** / **92.90** / **92.32** | **87.78** / **88.11** / **88.67** | 95.71 / **95.58** / **96.44**     | **95.15** / **96.59** / **97.27** | **90.80** / **92.00** / **94.20** |


## Cite this work

Chung-En Sun, Ge Yan, Tsui-Wei Weng, "ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models", arxiv preprint

```bibtex
@article{ThinkEdit,
   title={ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models},
   author={Sun, Chung-En and Yan, Ge and Weng, Tsui-Wei},
   journal={EMNLP},
   year={2025}
}
```

