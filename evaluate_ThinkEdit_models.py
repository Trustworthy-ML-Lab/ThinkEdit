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
from utils import model_dict, analyze_math_results, extract_questions, get_think_length

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

async def initialize_server_tracking():
    """Initialize the tracking dictionaries for server loads"""
    global server_loads, server_locks
    server_loads = {port: 0 for port in SERVER_PORTS}
    server_locks = {port: asyncio.Lock() for port in SERVER_PORTS}

async def get_next_server_url(endpoint: str="v1/chat/completions"):
    """
    Returns the URL of the server with the lowest number of pending requests.
    """
    global server_loads, server_locks
    
    # Find the server with minimum load
    min_load = float('inf')
    selected_port = None
    
    for port, load in server_loads.items():
        if load < min_load:
            min_load = load
            selected_port = port
    
    # Increment the load counter for the selected server
    async with server_locks[selected_port]:
        server_loads[selected_port] += 1
    
    return f"http://localhost:{selected_port}/{endpoint}"


async def query_llm_tokenizer_api(prompt: str, session: aiohttp.ClientSession, model: str) -> Dict:
    url = await get_next_server_url(endpoint="v1/tokenize")
    headers = {"Content-Type": "application/json"}
    data = {"model": model, "prompt": prompt}
    async with session.post(url, headers=headers, json=data) as response:
        return await response.json()
    

async def query_llm_completion_api_with_retry(prompt: str, session: aiohttp.ClientSession, model: str, 
                                 retry_count: int = 0) -> Dict:
    """
    Query the LLM API with retry mechanism and load balancing.
    """
    url = await get_next_server_url(endpoint="v1/completions")
    selected_port = int(url.split(':')[2].split('/')[0])  # Extract port from URL
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    try:
        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            # Decrement the load counter after request completes
            async with server_locks[selected_port]:
                server_loads[selected_port] -= 1
            return result
    except Exception as e:
        # Decrement the load counter if request fails
        async with server_locks[selected_port]:
            server_loads[selected_port] -= 1
            
        if retry_count < MAX_RETRIES:
            delay = min(BASE_DELAY * (2 ** retry_count) + random.uniform(0, 1), MAX_DELAY)
            print(f"Request failed on {url}: {e}. Retrying in {delay:.2f} seconds... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            await asyncio.sleep(delay)
            return await query_llm_completion_api_with_retry(prompt, session, model, retry_count + 1)
        else:
            print(f"Error querying LLM API after {MAX_RETRIES} retries: {e}")
            return None


async def query_llm_api_with_retry(question: str, session: aiohttp.ClientSession, model: str, instruction: str, 
                                 retry_count: int = 0, n_samples: int = 1) -> Dict:
    """
    Query the LLM API with retry mechanism and load balancing.
    """
    url = await get_next_server_url()
    selected_port = int(url.split(':')[2].split('/')[0])  # Extract port from URL
    
    headers = {"Content-Type": "application/json"}
    if instruction:
        question = f"Question: {question} {instruction}"
    data = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.6,
        "top_p": 0.95,
        "n": n_samples,
        "max_completion_tokens": MAX_RESPONSE_LENGTH
    }
    
    try:
        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            # Decrement the load counter after request completes
            async with server_locks[selected_port]:
                server_loads[selected_port] -= 1
    except Exception as e:
        # Decrement the load counter if request fails
        async with server_locks[selected_port]:
            server_loads[selected_port] -= 1
            
        if retry_count < MAX_RETRIES:
            delay = min(BASE_DELAY * (2 ** retry_count) + random.uniform(0, 1), MAX_DELAY)
            print(f"Request failed on {url}: {e}. Retrying in {delay:.2f} seconds... (Attempt {retry_count + 1}/{MAX_RETRIES})")
            await asyncio.sleep(delay)
            return await query_llm_api_with_retry(question, session, model, instruction, retry_count + 1, n_samples)
        else:
            print(f"Error querying LLM API after {MAX_RETRIES} retries: {e}")
            return None
    for choice in result["choices"]:
        if choice.finish_reason == "length": # If truncated due to max_completion_tokens
            completion_message = DEEPSEEK_THINK_TEMPLATE.format(q=question, i=instruction)
            completion_message = f"<think>{completion_message}\n</think>\n\nYeah, I think that's right.\n\n**Final Answer**\n"
            result = await query_llm_completion_api_with_retry(completion_message, session, model)
            choice["message"]["content"] = result["choices"][0]["message"]["content"]
        thinking_tokens = await query_llm_tokenizer_api(choice["message"]["reasoning_content"], session, model)
        choice["message"]["thinking_length"] = len(thinking_tokens["tokens"]) + 2
    return result


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
        sampling_params = SamplingParams(temperature=0.6,
                                         max_tokens=MAX_RESPONSE_LENGTH,
                                         top_p=0.95,
                                         n=n_samples,
                                         best_of=n_samples)
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

async def process_api_requests(questions: List[str], model: str, instruction: str, n_samples: int = 1) -> List[Dict]:
    """
    Process API requests asynchronously with load balancing.
    """
    # Initialize server tracking
    await initialize_server_tracking()
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def limited_query(question: str, index: int):
            async with semaphore:  # This limits concurrent requests
                await asyncio.sleep(REQUEST_DELAY)  # Add delay between requests
                return await query_llm_api_with_retry(question, session, model_dict[model], instruction, n_samples=n_samples)
        
        # Create tasks for all questions
        tasks = [
            limited_query(question, i)
            for i, question in enumerate(questions)
        ]
        
        # Process all tasks together while maintaining order
        responses = [None] * len(questions)
        failed_indices = []
        
        # Use gather to maintain order of responses
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results in order
        for i, result in enumerate(results):
            if isinstance(result, Exception) or result is None:
                failed_indices.append(i)
                responses[i] = None
            else:
                # Convert API response format to match our expected format
                samples = []
                for choice in result["choices"]:
                    samples.append({
                        "choices": [{
                            "message": choice["message"]
                        }]
                    })
                responses[i] = samples
        
        # Retry failed requests sequentially
        if failed_indices:
            print(f"\nRetrying {len(failed_indices)} failed requests sequentially...")
            for idx in failed_indices:
                question = questions[idx]
                for attempt in range(MAX_RETRIES):
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 2), MAX_DELAY * 2)
                    try:
                        response = await query_llm_api_with_retry(question, session, model_dict[model], instruction, n_samples=n_samples)
                        if response is not None:
                            # Convert API response format
                            samples = []
                            for choice in response["choices"]:
                                samples.append({
                                    "choices": [{
                                        "message": choice["message"]
                                    }]
                                })
                            responses[idx] = samples
                            print(f"Successfully retried request for question index {idx}")
                            break
                        await asyncio.sleep(delay)
                    except Exception as e:
                        print(f"Retry attempt {attempt + 1} failed for question index {idx}: {e}")
                        if attempt == MAX_RETRIES - 1:
                            print(f"All retries failed for question index {idx}")
        
        return responses

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
    parser.add_argument("--model", default="ThinkEdit-deepseek-qwen-1.5b", choices=["deepseek-qwen-14b", "deepseek-llama3-8b", "deepseek-qwen-1.5b", "ThinkEdit-deepseek-qwen-14b", "ThinkEdit-deepseek-llama3-8b", "ThinkEdit-deepseek-qwen-1.5b"]
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
