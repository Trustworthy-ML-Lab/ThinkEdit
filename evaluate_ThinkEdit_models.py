import json
import aiohttp
import argparse
import asyncio
from typing import List, Dict, Union
import time
import random
import os
import copy
from enum import Enum

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoConfig, GenerationConfig
from utils import model_dict, analyze_math_results, extract_questions, get_think_length, get_think_length_s1


# Add constants for retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1  # Base delay in seconds
MAX_DELAY = 10  # Maximum delay in seconds
DEEPSEEK_THINK_TEMPLATE = "<｜User｜>{q}{i}<｜Assistant｜>"
# Add new constants for rate limiting
REQUEST_DELAY = 0.1  # Delay between requests in seconds

# Add server configuration
current_port_index = 0


class InferenceMode(Enum):
    API = "api"
    OFFLINE = "offline"

# Add server load tracking
server_loads = {}
server_locks = {}

async def process_api_requests(questions: List[str], model: str, instruction: str, n_samples: int = 1) -> List[Dict]:
    """
    Process API requests asynchronously with load balancing.
    """
    pass
    
def query_llm_offline(questions: List[str], model: str, instruction: str,
                      n_samples: int = 1, tensor_parallel_size: int = 1) -> List[Dict]:
    """
    Run offline batch inference using vLLM directly.
    
    Args:
        questions: List of questions to process
        model: Name of the model to use
        n_samples: Number of samples to generate per question
        
    Returns:
        List of response dictionaries
    """
    try:
        # Initialize the LLM
        llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size,
                  max_model_len=MAX_RESPONSE_LENGTH+2048)
        tokenizer = llm.get_tokenizer()
        THINK_START_TOKEN_ID = tokenizer.encode("<think>", add_special_tokens=False)[0]
        THINK_END_TOKEN_ID = tokenizer.encode("</think>", add_special_tokens=False)[0]
        print(THINK_START_TOKEN_ID, THINK_END_TOKEN_ID)
        # Set sampling parameters
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        # Convert to GenerationConfig if needed
        gen_cfg = None
        try:
            gen_cfg = GenerationConfig.from_pretrained(model)
        except Exception:
            # Fallback: build from model config attributes
            gen_cfg = GenerationConfig(**{k: getattr(config, k) for k in ['temperature', 'top_k', 'top_p', 'repetition_penalty'] if hasattr(config, k)})

        sampling_params = SamplingParams(temperature=getattr(gen_cfg, 'temperature', 0.6),
                                         top_p=getattr(gen_cfg, 'top_p', 0.95),
                                         top_k=getattr(gen_cfg, 'top_k', None),
                                         repetition_penalty=getattr(gen_cfg, 'repetition_penalty', 1.0),
                                         max_tokens=MAX_RESPONSE_LENGTH,
                                         n=n_samples,
                                         best_of=n_samples,
                                         seed=random.randint(1, 1_000_000))
        continue_sampling_params = copy.deepcopy(sampling_params)
        continue_sampling_params.max_tokens = 256
        continue_sampling_params.n = 1
        continue_sampling_params.best_of = 1
        
        # Generate responses for all questions at once
        prompts = [DEEPSEEK_THINK_TEMPLATE.format(q=question, i=instruction) for question in questions]
        outputs = llm.generate(prompts, sampling_params)
        # Convert outputs to same format as API responses
        responses = []
        for prompt, output in zip(prompts, outputs):
            sample_responses = []
            for sample in output.outputs:
                think_length, has_think = get_think_length(sample.token_ids,
                                                        think_start_id=THINK_START_TOKEN_ID,
                                                        think_end_id=THINK_END_TOKEN_ID,
                                                        max_length=MAX_RESPONSE_LENGTH)
                if think_length >= MAX_RESPONSE_LENGTH:
                    sample.text += "\n</think>\n\nYeah, I think that's right.\n\n**Final Answer**\n"
                    continued_output = llm.generate(prompt + sample.text, continue_sampling_params)
                    sample.text += continued_output[0].outputs[0].text
                sample_responses.append({
                    "choices": [{
                        "message": {
                            "content": sample.text,
                            "thinking_length": think_length,
                            "reasoning_content": ""  # Offline mode doesn't separate reasoning
                        }
                    }]
                })
            responses.append(sample_responses)
        return responses
    
    except Exception as e:
        print(f"Error in offline inference: {e}")
        return [[None] * n_samples] * len(questions)

def process_responses(responses: List[Dict]) -> List[Dict]:
    """
    Extract relevant information from LLM responses.
    
    Args:
        responses: List of raw responses from the LLM
        
    Returns:
        List of processed responses with extracted information
    """
    processed = []
    for resp in responses:
        if resp is None:
            processed.append({
                "success": False,
                "error": "Failed to get response"
            })
            continue
            
        try:
            message = resp["choices"][0]["message"]
            processed.append({
                "success": True,
                "reasoning": message.get("reasoning_content", ""),
                "content": message.get("content", ""),
                "thinking_length": message.get("thinking_length", 0)
            })
        except (KeyError, IndexError) as e:
            processed.append({
                "success": False,
                "error": f"Failed to parse response: {e}"
            })
            
    return processed



async def async_main(dataset: str, mode: InferenceMode, model: str,
                     instruction: str, n_samples: int, tensor_parallel_size: int = 1):
    # Get questions from dataset
    questions = extract_questions(dataset)
    
    # Query LLM based on selected mode
    if mode == InferenceMode.API:
        # API mode - process requests asynchronously
        responses = await process_api_requests(questions, model, instruction, n_samples)
    else:
        # Offline mode - batch process all questions
        print("Running offline batch inference...")
        responses = query_llm_offline(questions, model_dict[model], instruction, n_samples,
                                      tensor_parallel_size=tensor_parallel_size)
    
    # Process responses for each sample
    processed_responses = [process_responses([resp[i] for resp in responses if resp is not None]) 
                         for i in range(n_samples)]

    # Save results
    results = {
        "questions": questions,
        "responses": processed_responses,
        "mode": mode.value,
        "n_samples": n_samples
    }
    
    # Save to file
    output_file = f"llm_responses_{dataset}_{mode.value}_{instruction}_{model}_samples{n_samples}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    stats, analyzed_results = analyze_math_results(processed_responses, dataset)
    print(stats)
    analyzed_results["instruction"] = instruction
    save_dir = f"ThinkEdit_model_evaluation_results/{dataset}/{model}/instruction_{instruction}"
    os.makedirs(save_dir, exist_ok=True)
    json.dump(analyzed_results, open(f"{save_dir}/results_samples{n_samples}.json", "w"), indent=4)

def main(dataset: str, mode: InferenceMode, model: str, instruction: str, n_samples: int,
         tensor_parallel_size: int = 1):
    """
    Entry point that runs the async main function.
    """
    asyncio.run(async_main(dataset, mode, model, instruction, n_samples, tensor_parallel_size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query LLM using API or offline batch inference")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "mmlu_elementary_math", "MATH-500", "MATH-level1", "MATH-level5"], 
                      help="Name of the dataset to process")
    parser.add_argument("--mode", choices=["api", "offline"], default="offline",
                      help="Inference mode: 'api' for local server API, 'offline' for batch inference")
    parser.add_argument("--model", default="ThinkEdit-deepseek-qwen-1.5b",
                      help="Name of the model to use")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--ports", type=int, nargs="+", default=[8000],
                      help="List of server ports to use (default: [8000])")
    parser.add_argument("--max_concurrent_requests", type=int, default=50,
                      help="Maximum number of concurrent requests (default: 50)")
    parser.add_argument("--n_samples", type=int, default=1,
                      help="Number of samples to generate per question (default: 1)")
    parser.add_argument("--max_length", type=int, default=16384,
                      help="Maximum length of the generated text (default: 16384)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                      help="Number of tensor parallel size (default: 1)")
    args = parser.parse_args()
    
    # Set global SERVER_PORTS from command line argument
    SERVER_PORTS = args.ports
    MAX_CONCURRENT_REQUESTS = args.max_concurrent_requests
    MAX_RESPONSE_LENGTH = args.max_length
    main(args.dataset, InferenceMode(args.mode), args.model, args.instruction,
         args.n_samples, args.tensor_parallel_size)
