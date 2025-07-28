import json
import random
from loguru import logger
from utils import load_jsonlines, write_jsonlines
from instagger_analysis import get_instagger_tags, InferenceConfig
from transformers import AutoTokenizer
from typing import List, Dict


# use instagger to get tags
def filter_data_by_length_percentile(sampled_data: List[dict], tokenizer, lower_percentile=0.02, upper_percentile=0.98) -> List[dict]:
    """Filter data by removing examples with instruction/response lengths outside percentile range.
    
    Args:
        sampled_data: List of instruction/response pairs
        tokenizer: Tokenizer for calculating token lengths
        lower_percentile: Lower percentile cutoff (default 0.02)
        upper_percentile: Upper percentile cutoff (default 0.98)
        
    Returns:
        Filtered list of examples within length bounds
    """
    # Batch tokenize instructions and responses
    instructions = [item['instruction'] for item in sampled_data]
    responses = [item['response'] for item in sampled_data]
    instruction_tokens = tokenizer(instructions, add_special_tokens=False)['input_ids']
    response_tokens = tokenizer(responses, add_special_tokens=False)['input_ids']
    instruction_lengths = [len(tokens) for tokens in instruction_tokens]
    response_lengths = [len(tokens) for tokens in response_tokens]

    # Calculate percentile boundaries
    instruction_lower = sorted(instruction_lengths)[int(len(instruction_lengths) * lower_percentile)]
    instruction_upper = sorted(instruction_lengths)[int(len(instruction_lengths) * upper_percentile)]
    response_lower = sorted(response_lengths)[int(len(response_lengths) * lower_percentile)]
    response_upper = sorted(response_lengths)[int(len(response_lengths) * upper_percentile)]

    # Filter data
    filtered_data = []
    for item, inst_len, resp_len in zip(sampled_data, instruction_lengths, response_lengths):
        if (instruction_lower <= inst_len <= instruction_upper and 
            response_lower <= resp_len <= response_upper):
            item['instruction_tokens'] = inst_len
            item['response_tokens'] = resp_len
            filtered_data.append(item)

    logger.info(f"Filtered data size: {len(filtered_data)} (removed {len(sampled_data)-len(filtered_data)} examples)")
    return filtered_data


def calculate_tag_values(data: List[dict], tags_list: List[List[dict]], tokenizer) -> List[dict]:
    """Calculate value for each tag based on response token length divided by number of tags.
    
    Args:
        data: List of instruction/response pairs
        tags_list: List of tag lists for each instruction
        tokenizer: Tokenizer for calculating response lengths
        
    Returns:
        List of dicts containing tag stats sorted by average value
    """
    tag_values = {}
    tag_counts = {}
    
    for instruction, tags in zip(data, tags_list):
        response_tokens = len(tokenizer.tokenize(instruction['response']))
        
        if not tags or response_tokens == 0:
            continue
        value_per_tag = response_tokens / len(tags)
        for tag in tags:
            tag_name = tag['tag']
            if tag_name not in tag_values:
                tag_values[tag_name] = 0
                tag_counts[tag_name] = 0
            tag_values[tag_name] += value_per_tag
            tag_counts[tag_name] += 1

    result_tags = [
        {
            'tag': tag,
            'count': tag_counts[tag],
            'total_value': tag_values[tag],
            'average_value': tag_values[tag] / tag_counts[tag],
        }
        for tag in tag_values
    ]
    return sorted(result_tags, key=lambda x: x['average_value'], reverse=True)

import os
def main():
    inference_config = InferenceConfig()
    tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_path = "/home/zhe/tag-instruct-experiment/build_reward_model/dataset/sft/alpaca/alpaca_data_cleaned.jsonl"
    logger.info(f"Loading data from {data_path}")
    data = load_jsonlines(data_path)

    # Split data into 5 chunks
    chunk_size = len(data) // 5
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    logger.info(f"Split data into {len(chunks)} chunks of size ~{chunk_size}")

    all_filtered_data = []
    all_tags_list = []
    for i, chunk in enumerate(chunks):
        if os.path.exists(f"./alpaca_filtered_data_{i}.jsonl"):
            logger.info(f"Skipping chunk {i+1}/{len(chunks)} because it already exists")
            all_filtered_data.extend(load_jsonlines(f"./alpaca_filtered_data_{i}.jsonl"))
            all_tags_list.extend(load_jsonlines(f"./alpaca_chunk_tags_{i}.jsonl"))
            continue
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        filtered_chunk = filter_data_by_length_percentile(chunk, tokenizer)
        chunk_tags_list = get_instagger_tags(filtered_chunk)
        all_filtered_data.extend(filtered_chunk)
        all_tags_list.extend(chunk_tags_list)
        
        write_jsonlines(filtered_chunk, f"./alpaca_filtered_data_{i}.jsonl")
        write_jsonlines(chunk_tags_list, f"./alpaca_chunk_tags_{i}.jsonl")
        
    # 计算tag的value
    tag_values = calculate_tag_values(all_filtered_data, all_tags_list, tokenizer)
    output_path = "./alpaca_tag_value.jsonl"
    write_jsonlines(tag_values, output_path)
    logger.info(f"Saved {len(tag_values)} tag values to {output_path}")

   

if __name__ == "__main__":
    main()
