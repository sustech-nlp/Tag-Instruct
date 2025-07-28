import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from loguru import logger
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import load_jsonlines, write_jsonlines
from vllm_client import parallel_inference
import re
from tag_instruct import * 
from tqdm import tqdm, trange
import os

@dataclass
class Config:
    input_files: List[str] = field(default_factory=lambda: [
        # Add more input files here
    ])
                                   
    tag_data_path: str = "/home/zhe/tag-instruct-experiment/wizardlm_tag_value.jsonl"
    model_name: str = "/data/zhe/mistralai--Ministral-8B-Instruct-2410"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_candidates: int = 20  # Number of tag candidates to generate
    output_path: str = "./tag_pairs_combined_wizardlm.jsonl"  # Changed to reflect combined output


def parsing_explan_tag(prompt: str) -> Tuple[str, str]:
    explan_parttern = r'\[Explanation\]\s*(.*)\s*\[New Tag\]'
    new_tag_pattern = r'\[New Tag\]\s*(.*)'
    new_tag = re.search(new_tag_pattern, prompt, re.DOTALL)
    new_explan = re.search(explan_parttern, prompt, re.DOTALL)
    
    if not new_tag or not new_explan:
        logger.warning(f"Failed to parse response: {prompt}")
    
    return [
        new_explan.group(1).strip() if new_explan else "None",
        new_tag.group(1).strip() if new_tag else "None"
    ]
    

class TagRewardModel:
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
    
        tags_data = load_jsonlines(self.config.tag_data_path)
        filtered_tags = [tag for tag in tags_data if tag['count'] > 2]
        logger.info(f"Tags count - before filtering: {len(tags_data)}, after filtering count>2: {len(filtered_tags)}")
        sorted_tags = sorted(filtered_tags, key=lambda x: x['average_value'], reverse=True)
        top_tags = sorted_tags[:1000]
        bottom_tags = sorted_tags[-1000:]
        return top_tags, bottom_tags

    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text)

    def calculate_reward(self, candidate_tag: str, top_tags: List[Dict], bottom_tags: List[Dict]) -> float:
        tag_embedding = self.get_embedding(candidate_tag)
        
        # Calculate similarity with top tags
        top_similarities = []
        for tag in top_tags:
            tag_embedding_top = self.get_embedding(tag['tag'])
            similarity = np.dot(tag_embedding, tag_embedding_top) / (
                np.linalg.norm(tag_embedding) * np.linalg.norm(tag_embedding_top)
            )
            top_similarities.append(similarity)
            
        # Calculate similarity with bottom tags
        bottom_similarities = []
        for tag in bottom_tags:
            tag_embedding_bottom = self.get_embedding(tag['tag'])
            similarity = np.dot(tag_embedding, tag_embedding_bottom) / (
                np.linalg.norm(tag_embedding) * np.linalg.norm(tag_embedding_bottom)
            )
            bottom_similarities.append(similarity)
            
        # Reward is average similarity with top tags minus average similarity with bottom tags
        reward = np.mean(top_similarities) - np.mean(bottom_similarities)
        return reward

    def calculate_reward_batch(self, candidate_tags: List[str], top_tags: List[Dict], bottom_tags: List[Dict]) -> List[float]:
        """
        Calculate rewards for multiple candidate tags in batch.
        
        Example usage:
            model = TagRewardModel(config)
            
            # Example inputs
            candidate_tags = ["machine_learning", "data_science", "neural_networks"] 
            top_tags = [
                {"tag": "artificial_intelligence", "count": 10, "average_value": 0.9},
                {"tag": "deep_learning", "count": 8, "average_value": 0.85}
            ]
            bottom_tags = [
                {"tag": "random_tag", "count": 3, "average_value": 0.1},
                {"tag": "irrelevant_tag", "count": 4, "average_value": 0.2}
            ]
            
            # Calculate rewards for all candidates at once
            rewards = model.calculate_reward_batch(candidate_tags, top_tags, bottom_tags)
            # Returns something like: [0.75, 0.68, 0.71]
            # Higher rewards indicate tags more similar to top_tags and less similar to bottom_tags
        """
        # Pre-compute embeddings for top and bottom tags
        top_embeddings = self.embedding_model.encode([tag['tag'] for tag in top_tags])
        bottom_embeddings = self.embedding_model.encode([tag['tag'] for tag in bottom_tags])
        
        # Process candidate tags in chunks to avoid memory issues
        CHUNK_SIZE = 1000
        rewards = []
        
        for i in trange(0, len(candidate_tags), CHUNK_SIZE):
            chunk_candidates = candidate_tags[i:i+CHUNK_SIZE]
            candidate_embeddings = self.embedding_model.encode(chunk_candidates)
            
            # Calculate similarities with top tags
            top_similarities = np.dot(candidate_embeddings, top_embeddings.T) / (
                np.linalg.norm(candidate_embeddings, axis=1)[:, np.newaxis] *
                np.linalg.norm(top_embeddings, axis=1)
            )
            
            # Calculate similarities with bottom tags
            bottom_similarities = np.dot(candidate_embeddings, bottom_embeddings.T) / (
                np.linalg.norm(candidate_embeddings, axis=1)[:, np.newaxis] *
                np.linalg.norm(bottom_embeddings, axis=1)
            )
            
            # Calculate rewards for this chunk
            chunk_rewards = np.mean(top_similarities, axis=1) - np.mean(bottom_similarities, axis=1)
            rewards.extend(chunk_rewards.tolist())
            
        return rewards

    def generate_tag_pairs(self, instructions: List[str], tags_list: List[str]):
        """Generate accepted/rejected tag pairs using BON and reward model"""
        
        # Generate prompts for each instruction
        prompt_list = []
        for instruction, tags in zip(instructions, tags_list):
            prompt = add_tags_only_template(tags, instruction) 
            prompt_list.extend([prompt] * self.config.num_candidates)
        
        # Pass the model_name to parallel_inference
        responses = parallel_inference(
            prompt_list, 
            temperature=1, 
            top_p=0.95,
            model_name_or_path=self.config.model_name
        ) 
        
        # Extract base filename without extension and add responses_ prefix
        base_filename = os.path.splitext(os.path.basename(self.config.input_files[0]))[0]
        response_file = f"./responses_{base_filename}.jsonl"
        write_jsonlines([{"responses": response} for response in responses], response_file)

        explanations_tags_list = [parsing_explan_tag(response) for response in responses]
        all_explanations = [item[0] for item in explanations_tags_list] #size: 10000
        all_tags = [item[1] for item in explanations_tags_list] #size: 10000
        
    
        # Calculate rewards for all tags at once
        if all_tags:
            all_rewards = self.calculate_reward_batch(all_tags, self.top_tags, self.bottom_tags)
        else:
            return []
        
    
        
        # Group candidates by instruction based on num_candidates
        preference_results = []
        num_instructions = len(instructions)

        for i in range(num_instructions):
            start_idx = i * self.config.num_candidates
            end_idx = start_idx + self.config.num_candidates
            
            # Get candidates for current instruction
            instruction_explanations = all_explanations[start_idx:end_idx]
            instruction_tags = all_tags[start_idx:end_idx]
            instruction_rewards = all_rewards[start_idx:end_idx]
            
            if len(instruction_tags) >= 2:
                candidates = list(zip(instruction_explanations, instruction_tags, instruction_rewards))
                sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True) 
                
                valid_candidates = [c for c in sorted_candidates if c[0] != "None" and c[1] != "None"]
                best = valid_candidates[0] if len(valid_candidates) >= 2 else sorted_candidates[0]
                worst = valid_candidates[-1] if len(valid_candidates) >= 2 else sorted_candidates[-1]
                
                # Create preference pair
                preference_results.append({
                    'instruction': add_tags_only_template(tags_list[i], instructions[i]),
                    'accepted_response': f'[Explanation] {best[0]}\n[New Tag] {best[1]}',
                    'rejected_response': f'[Explanation] {worst[0]}\n[New Tag] {worst[1]}'
                })
                
        return preference_results


def convert_to_dpo_format(preference_results: List[Dict]) -> List[Dict]:
    """Convert tag pairs data to DPO format for training."""
    dpo_data = []
    for item in preference_results:
        try:
            conversation = [
                {"from": "human", "value": item["instruction"]}
            ]
            
            dpo_item = {
                "conversations": conversation,
                "chosen": {
                    "from": "gpt",
                    "value": item["accepted_response"]
                },
                "rejected": {
                    "from": "gpt", 
                    "value": item["rejected_response"]
                }
            }
            dpo_data.append(dpo_item)
        except KeyError as e:
            logger.warning(f"Missing key in data item: {e}")
            continue
    
    return dpo_data

if __name__ == "__main__":
    config = Config()
    model = TagRewardModel(config)
    model.top_tags, model.bottom_tags = model.load_data()
    
    # Process multiple input files
    all_preference_results = []
    for input_file in config.input_files:
        data = load_jsonlines(input_file)
        data = get_tags(data)
        instructions = [item['instruction'] for item in data]
        tags_list = [item['tag'] for item in data]
        
        # Generate preferences for current file
        preference_results = model.generate_tag_pairs(instructions, tags_list)
        all_preference_results.extend(preference_results)
        
        logger.info(f"Processed {input_file}, generated {len(preference_results)} preference pairs")
    
    # Write combined results in original format
    write_jsonlines(all_preference_results, config.output_path)
    logger.info(f"Total preference pairs generated: {len(all_preference_results)}")

    # Convert and save in DPO format
    dpo_output_path = config.output_path.replace('.jsonl', '_dpo.jsonl')
    dpo_data = convert_to_dpo_format(all_preference_results)
    write_jsonlines(dpo_data, dpo_output_path)
    logger.info(f"Saved {len(dpo_data)} items in DPO format to {dpo_output_path}")

    

    
    
    

    