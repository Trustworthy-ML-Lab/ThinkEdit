import os
import json
import re
import math
import numpy as np
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from math_grader import math_equal, strip_string

DATASET_MAP = {
    "gsm8k": {"args": ("openai/gsm8k", "main"), "question_key": "question"},
    "MATH-500": {"args": ("HuggingFaceH4/MATH-500",), "question_key": "problem"},
    "mmlu_elementary_math": {"args": ("cais/mmlu", "elementary_mathematics"), "question_key": "prompt"},
    "MATH-level1": {"args": ("EleutherAI/hendrycks_math",), "question_key": "problem"},
    "MATH-level5": {"args": ("EleutherAI/hendrycks_math",), "question_key": "problem"}

}

model_dict = {
    "deepseek-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-llama3-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "ThinkEdit-deepseek-qwen-1.5b": "cesun/ThinkEdit-deepseek-qwen-1.5b",
    "ThinkEdit-deepseek-llama3-8b": "cesun/ThinkEdit-deepseek-llama3-8b",
    "ThinkEdit-deepseek-qwen-14b": "cesun/ThinkEdit-deepseek-qwen-14b",
    "ThinkEdit-deepseek-qwen-32b": "cesun/ThinkEdit-deepseek-qwen-32b"
    }

def get_think_length(output_ids, think_start_id, think_end_id, max_length=8192):
    think_starts = [i for i, token in enumerate(output_ids) if token == think_start_id]
    think_ends = [i for i, token in enumerate(output_ids) if token == think_end_id]
    
    if think_starts and think_ends:
        return think_ends[0] - think_starts[0] + 1, True
    elif think_starts and not think_ends:
        return max_length, False
    elif not think_starts and think_ends:
        return think_ends[0] + 1, False
    else:
        return -1, False

def find_sublist_indices(lst, sublist):
    """Find all start indices where sublist appears in lst"""
    sub_len = len(sublist)
    return [i for i in range(len(lst) - sub_len + 1) if lst[i:i + sub_len] == sublist]

def get_think_length_s1(output_ids, think_start_id, think_end_id, max_length=8192):
    think_starts = find_sublist_indices(output_ids, think_start_id)
    think_ends = find_sublist_indices(output_ids, think_end_id)

    if think_starts and think_ends:
        return think_ends[0] + len(think_end_id) - think_starts[0], True
    elif think_starts and not think_ends:
        return max_length, False
    elif not think_starts and think_ends:
        return think_ends[0] + len(think_end_id), False
    else:
        return -1, False

def extract_questions(dataset):
    """
    Loads the specified dataset (possibly filtering by MATH 'Level 1' or 'Level 5'),
    then returns the relevant question column.
    """

    # Special handling for MATH-level1 and MATH-level5
    if dataset in ["MATH-level1", "MATH-level5"]:
        # Identify the target level (string in 'level' column)
        target_level = "Level 1" if dataset == "MATH-level1" else "Level 5"

        # Get all subsets (config names) for "EleutherAI/hendrycks_math"
        subsets = get_dataset_config_names("EleutherAI/hendrycks_math")

        # Load and filter all test subsets
        filtered_subsets = []
        for subset_name in subsets:
            # Load the test split for each subset
            ds_subset = load_dataset("EleutherAI/hendrycks_math", subset_name, split="test")

            # Filter to keep rows with the desired 'level'
            ds_subset = ds_subset.filter(lambda x: x["level"] == target_level)

            # Append if not empty
            if len(ds_subset) > 0:
                filtered_subsets.append(ds_subset)

        # Concatenate all filtered subsets
        if len(filtered_subsets) == 0:
            # Handle edge case: no data found
            return []
        d = concatenate_datasets(filtered_subsets)

    else:
        # For all other datasets, load normally
        d = load_dataset(*DATASET_MAP[dataset]["args"], split="test")

    # Example: If it's the mmlu_elementary_math dataset, format it
    if dataset == "mmlu_elementary_math":
        def format_prompt(example):
            prompt = f"{example['question']}\n"
            for i, choice in enumerate(example["choices"]):
                prompt += f"{i+1}. {choice}\n"
            prompt += "Choose from the above options and the answer format should be '\\boxed{{index}}'."
            return {"prompt": prompt}

        d = d.map(format_prompt)

    # Finally, return the relevant question column
    return d[DATASET_MAP[dataset]["question_key"]]

def extract_answer(text):
    if text is None:
        return None
    # Step 1: Remove everything that is not a number, letter, ".", or "-"
    # text = re.sub(r'[^0-9a-zA-Z{}\\.\-]', '', text)
    # Try extracting from 'boxed' first
    boxed_matches = extract_boxed(text)
    if boxed_matches:
        extracted_answer = boxed_matches[-1][1:-1]
        return strip_string(extracted_answer)

    # Fallback: extract any numbers
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    if not numbers:
        return None

    try:
        extracted_number = float(numbers[-1])
        # Guard against infinity
        if math.isinf(extracted_number):
            return None
        
        return numbers[-1]
    except (ValueError, OverflowError):
        return None

def extract_boxed(text):
    pattern = re.compile(r'boxed\{')
    matches = []
    stack = []
    
    i = 0
    while i < len(text):
        match = pattern.search(text, i)
        if not match:
            break
        
        start = match.end() - 1  # Position at the first `{`
        stack.append(start)
        i = start + 1
        count = 1  # To track `{}` pairs
        
        while i < len(text) and stack:
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:  # Found a matching closing `}`
                    start = stack.pop()
                    matches.append(text[start:i+1])
                    break
            i += 1
    
    return matches


def analyze_math_results(responses, dataset_name, extractor=extract_answer):
    """
    Analyze results for multiple samples per question.
    
    Args:
        responses: List of lists, where each inner list contains responses for one sample
        dataset_name: Name of the dataset
        extractor: Function to extract answers from responses
    """
    if dataset_name in ["MATH-level1", "MATH-level5"]:
        # Identify the target level (string in 'level' column)
        target_level = "Level 1" if dataset_name == "MATH-level1" else "Level 5"

        # Get all subsets (config names) for "EleutherAI/hendrycks_math"
        subsets = get_dataset_config_names("EleutherAI/hendrycks_math")

        # Load and filter all test subsets
        filtered_subsets = []
        for subset_name in subsets:
            # Load the test split for each subset
            ds_subset = load_dataset("EleutherAI/hendrycks_math", subset_name, split="test")

            # Filter to keep rows with the desired 'level'
            ds_subset = ds_subset.filter(lambda x: x["level"] == target_level)

            # Append if not empty
            if len(ds_subset) > 0:
                filtered_subsets.append(ds_subset)

        # Concatenate all filtered subsets
        if len(filtered_subsets) == 0:
            # Handle edge case: no data found
            return []
        dataset = concatenate_datasets(filtered_subsets)

    else:
        # For all other datasets, load normally
        dataset = load_dataset(*DATASET_MAP[dataset_name]["args"], split="test")
    
    # Get ground truth answers
    if dataset_name == "gsm8k":
        answers = [str(ex['answer']).split('####')[-1].strip() for ex in dataset]
    elif dataset_name == "MATH-level1" or dataset_name == "MATH-level5":
        answers = [extract_answer(ex['solution']) for ex in dataset]
    elif "mmlu" in dataset_name:
        answers = [str(ex['answer']+1) for ex in dataset]
    else:
        answers = dataset['answer']
    answers = [strip_string(a) for a in answers]
    
    # Process each sample
    all_stats = []
    for sample_responses in responses:
        response_texts = [resp['content'] for resp in sample_responses]
        thinking_texts = [resp['reasoning'] for resp in sample_responses]
        thinking_lengths = [resp['thinking_length'] for resp in sample_responses]
        
        # Extract predictions for this sample
        predicted = [extractor(resp) for resp in response_texts]
        
        # Compare predictions to ground truth
        correctness = []
        for pred, ans in zip(predicted, answers):
            if pred is None:
                correctness.append(0)
            else:
                try:
                    correctness.append(int(math_equal(pred, ans)))
                except:
                    correctness.append(0)
        
        sample_stats = {
            'accuracy': np.mean(np.array(correctness)),
            'avg_thinking_length': np.mean(thinking_lengths),
            'think_lengths': thinking_lengths,
            'response_texts': response_texts,
            'correctness': correctness,
            'predicted': predicted,
        }
        all_stats.append(sample_stats)
    
    # Calculate aggregate statistics
    aggregate_stats = {
        'accuracy': np.mean([stats['accuracy'] for stats in all_stats]),
        'avg_thinking_length': np.mean([stats['avg_thinking_length'] for stats in all_stats]),
    }
    
    analyzed_results = {
        "sample_results": all_stats,
        "answers": answers,
        "aggregate_stats": aggregate_stats,
    }
    
    return aggregate_stats, analyzed_results