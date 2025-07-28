from copy import deepcopy

from numpy import save
from inference.vllm_client import parallel_inference_instagger
from dataclasses import dataclass
import re
from typing import List
from loguru import logger 
import os
from utils import load_json, write_jsonlines
import json
from transformers import AutoTokenizer
from matplotlib import pyplot as plt




@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    model_name_or_path: str =  "/home/zhe/models/lukeminglkm/instagger_llama2"


def extract_tags(input_string):
    pattern = r'"tag":\s*"([^"]*)",\s*"explanation":\s*"([^"]*)"'
    matches = re.findall(pattern, input_string)
    return [{"tag": tag if tag else None, "explanation": explanation if explanation else None} 
            for tag, explanation in matches]

inference_config = InferenceConfig()

def get_tags(data: List[dict]) -> List[dict]:
    prompts = [entry["instruction"] for entry in data]
    # prompts = [get_tags_template(instruction) for instruction in instructions]
    responses = parallel_inference_instagger(prompts, max_tokens=1024, **vars(inference_config))
    for i, response in enumerate(responses):
        data[i]["instags"] = response
    return data

def get_instagger_tags(data: List[dict]) -> List[List[dict]]:
    data = get_tags(data)
    tags_str = [entry["instags"] for entry in data]
    tags_list = []
    for tags in tags_str:
        tags_list.append(extract_tags(tags))
    return tags_list


def get_experiment_tags(data: List[dict]) -> List[List[dict]]:
    data = get_tags(data)
    tags_str = [entry["instags"] for entry in data]
    tags_list = []
    for tags in tags_str:
        tags_list.append(extract_tags(tags))
        
        
    # data['tags'] = tags_list
    for i, entry in enumerate(data):    
        tag_str = ",".join([tag['tag'] for tag in tags_list[i]])
        entry['tag'] = tag_str
    return data

        
        

def get_complexity_diversity(data: List[dict]) -> List[dict]:
    
    data = get_tags(data)
    tags_str = [entry["instags"] for entry in data]
    tags_list = []
    for tags in tags_str:
        tags_list.append(extract_tags(tags))
    # 计算复杂性
    complexity = [len(tag) for tag in tags_list]
    avg_complexity = sum(complexity) / len(complexity) if len(complexity) > 0 else 0
    
    # 计算多样性
    try:
        diversity = len(set(tag["tag"] for tags in tags_list for tag in tags))
    except KeyError as e:
        logger.error(f"Missing 'tag' key in one of the tags. Error: {e}")
        diversity = 0
    
    # 输出调试信息
    logger.debug(f"complexity: {complexity}")
    logger.debug(f"avg_complexity: {avg_complexity}")
    logger.debug(f"diversity: {diversity}")
    
    return avg_complexity, diversity, len(data)

def calculate_average_tokens(data: List[dict], tokenizer) -> tuple[float, float]:
    if 'instruction' not in data[0] or 'response' not in data[0]:
        raise ValueError("The data does not contain 'instruction' or 'response' key.")
    
    instruction_token_counts = []
    response_token_counts = []
    for entry in data:
        instruction = entry["instruction"]
        response = entry["response"]
        instruction_tokens = tokenizer.tokenize(instruction)
        response_tokens = tokenizer.tokenize(response)
        instruction_token_counts.append(len(instruction_tokens))
        response_token_counts.append(len(response_tokens))
    avg_instruction_tokens = sum(instruction_token_counts) / len(instruction_token_counts) if instruction_token_counts else 0
    avg_response_tokens = sum(response_token_counts) / len(response_token_counts) if response_token_counts else 0
    return avg_instruction_tokens, avg_response_tokens


def get_infomation_request(data: List[dict]) -> List[dict]:
    tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name_or_path)
    avg_complexity, diversity, num_files = get_complexity_diversity(data)
    avg_instruction_tokens, avg_response_tokens = calculate_average_tokens(data, tokenizer)
    return avg_complexity, diversity, avg_instruction_tokens, avg_response_tokens



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name_or_path)
 
    file_path_list = [
        #TODO: add file path here
    ]
    
    
    complexities = []
    diversities = []
    avg_tokens_list = []
    from tqdm import tqdm

    results = []
    output_file = "./analysis_results.jsonl"
    
    for filepath in tqdm(file_path_list):
        data = load_json(filepath)
        avg_complexity, diversity, avg_instruction_tokens, avg_response_tokens = get_infomation_request(data)
        
        # Extract name from filepath
        name = os.path.basename(filepath).split('.')[0]
        
        result = {
            "name": name,
            "complexity": avg_complexity,
            "diversity": diversity, 
            "instruction_token": avg_instruction_tokens,
            "response_token": avg_response_tokens,
            "num_files": len(data)
        }
        results.append(result)
        
        # Save result immediately after processing each file
        with open(output_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        logger.debug(f"File: {name}")
        logger.debug(f"Average Complexity: {avg_complexity}")
        logger.debug(f"Diversity: {diversity}")
        logger.debug(f"Average Instruction Tokens: {avg_instruction_tokens}")
        logger.debug(f"Average Response Tokens: {avg_response_tokens}")
        logger.debug(f"Number of Files: {len(data)}")
        logger.debug(f"Result saved to {output_file}")

    print(results)
    

