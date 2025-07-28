import ray
from json import load
from typing import List, Dict
import numpy as np
from vllm import LLM, SamplingParams
from scipy.special import softmax
import os
from loguru import logger
import re
import json

# 设置 GPU 数量
gpus = [ "2", "3", "4", "5", "6", "7"]
num_gpus = len(gpus)
print(f"Using {num_gpus} GPUs: {gpus}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_jsonlines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonlines(data, file_path):
    if os.path.exists(file_path):
        logger.info(f"Skipping operation because {file_path} already exists.")
        return
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

config_table = {
    "llama2": {
        "max_model_len": 2048,
        "id2score": {29900: "0", 29896: "1"}
    },
    "llama3": {
        "max_model_len": 8192,
        "id2score": {15: "0", 16: "1"}
    },
    "mistral": {
        "max_model_len": 2000,
        "id2score": {28734: "0", 28740: "1"}
    }
}

def get_model_config(model_name_or_path):
    for key in config_table:
        if key in model_name_or_path.lower():
            logger.info(f"Using config for {key}")
            return config_table[key]
    return config_table["mistral"]

def parser_score(input_list: List[str]) -> List[int]:
    pattern = re.compile(r'score:\s*(\d)', re.IGNORECASE)
    scores = [int(match.group(1)) if (match := pattern.search(s)) else 0 for s in input_list]
    return scores

@ray.remote(num_gpus=1)
def vllm_inference(model_name_or_path: str, input_data: List[str], gpu_id: str, max_tokens: int = 256, temperature: float = 0, top_p: float = 0.9, skip_special_tokens: bool = True):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    logger.info(f"Process running on GPU {gpu_id}")
    
    config = get_model_config(model_name_or_path)
    llm = LLM(model=model_name_or_path, tokenizer_mode="auto", trust_remote_code=True, max_model_len=config["max_model_len"], gpu_memory_utilization=0.90)
    
    if "llama3" in model_name_or_path:
        tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=skip_special_tokens,
                                         stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("")])
    else:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, skip_special_tokens=skip_special_tokens)
    
    outputs = llm.generate(input_data, sampling_params)
    return [output.outputs[0].text for output in outputs]

@ray.remote(num_gpus=1)
def vllm_logprobs(model_name_or_path: str, input_data: List[str], gpu_id: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    logger.info(f"Process running on GPU {gpu_id}")
    llm = LLM(model_name_or_path, max_num_batched_tokens=4096)
    sampling_params = SamplingParams(max_tokens=2, logprobs=20) # from deita
    outputs = llm.generate(input_data, sampling_params) 
    return [output.outputs[0].logprobs[0] for output in outputs]



def parallel_inference(prompt_list: List[Dict], model_name_or_path: str, max_tokens: int = 256, temperature: float = 0, top_p: float = 0.9, skip_special_tokens: bool = True, score=False):
    ray.init()
    chunk_size = len(prompt_list) // num_gpus
    chunks = [prompt_list[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)]
    if len(prompt_list) % num_gpus != 0:
        chunks[-1].extend(prompt_list[num_gpus * chunk_size:])
    
    # 提交 Ray 任务，分配到各个 GPU 上运行
    futures = [vllm_inference.remote(model_name_or_path, chunks[i], gpus[i], max_tokens, temperature, top_p, skip_special_tokens) for i in range(num_gpus)]
    
    # 获取结果
    results = ray.get(futures)
    flat_results = [item for sublist in results for item in sublist]
    ray.shutdown()
    
    return flat_results if not score else parser_score(flat_results)


def parallel_inference_logprobs(prompt_list: List[Dict], model_name_or_path: str):
    ray.init()
    chunk_size = len(prompt_list) // num_gpus
    chunks = [prompt_list[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)]
    if len(prompt_list) % num_gpus != 0:
        chunks[-1].extend(prompt_list[num_gpus * chunk_size:])
    
    # 提交 Ray 任务，分配到各个 GPU 上运行
    futures = [vllm_logprobs.remote(model_name_or_path, chunks[i], gpus[i]) for i in range(num_gpus)]
    
    # 获取结果
    results = ray.get(futures)
    flat_results = [item for sublist in results for item in sublist]
    ray.shutdown()
    
    return flat_results





if __name__ == "__main__":
    ray.init()
    
    model_name_or_path = "/home/admin/data/huggingface_model/Qwen2.5-Math-7B-Instruct"
    
    get_response_template = '''Below is an instruction that describes a task, write a response that appropriately completes the request. Please reason step by step, and put your final answer within \\boxed{{}}.
    
Example:
### Instruction:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? 

### Response:
Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. So the answer is \\boxed{{72}}.

Your Task:
### Instruction:
{instruction}

### Response: 
'''

    pth = "/home/admin/Tag-instruct/result/tag_instruct_math_very_very_big/response_data_mistral.jsonl"
    otuput_pth = "/home/admin/Tag-instruct/result/tag_instruct_math_very_very_big/response_data_qwen2.jsonl"
    
    data = load_jsonlines(pth)
    prompts = [get_response_template.format(instruction=entry["instruction"]) for entry in data]
    
    result = parallel_inference(prompts, model_name_or_path, max_tokens=1024, temperature=0, top_p=0.95, skip_special_tokens=True, score=False)
    
    for i, entry in enumerate(data):
        entry["response"] = result[i]
    
    write_jsonlines(data, otuput_pth)
    
    ray.shutdown()