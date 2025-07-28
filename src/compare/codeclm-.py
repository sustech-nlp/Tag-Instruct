from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger
from inference.vllm_client import parallel_inference
import random
import re
from utils import * 
import random
random.seed(42)


# README
# 1.是对codeclm的改进, 去除了过滤操作, 多次迭代
# 3. 没有instruction_cleaning 操作

@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    model_name_or_path: str = "/home/zhe/.cache/modelscope/hub/qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
    
    
    
    
    
def extract_metadata(instruction: str) -> str:
    prompt = """I want you to act as an instruction analyzer.
Given an instruction, you should recognize its use case and the skills (or knowledge) required for a large language model (LLM) to answer the question.
Generate the use case and skills required without any explanation.
List at most 3 skills, each skill should be transferable, so that LLM can leverage them to answer similar questions.
Avoid using "skill", "knowledge" to describe a skill, and each skill should be concise (2-3 words).
Follow the examples below to analyze the given instruction.

#Example 1#
As a sports commentator, describe the winning play in the final seconds of a championship game.
Output:
Task: creative writing
Skills: role-play, sports

#Example 2#
How to read a large file (> 2T) using python?
Output:
Task: code generation
Skills: python

#Example 3#
The method section of your paper is too brief and does not explain how your proposed model works in detail. How can you provide more details of the hierarchical encoder and the cascaded selectors, such as their architectures, inputs, outputs, and parameters?
Output:
Task: general knowledge question answering
Skills: academic writing, machine learning

Your Task:
{instruction}
Output:
"""
    return prompt.format(instruction=instruction)

def parse_metadata(metadata_str: str) -> Tuple[str, List[str]]:
    try:
        # 只匹配 Task: 和 Skills: 格式
        task_match = re.search(r'task:\s*([^\n]+)', metadata_str.lower())
        skills_match = re.search(r'skills:\s*([^\n]+)', metadata_str.lower())
        
        if not task_match or not skills_match:
            logger.warning(f"Could not parse metadata format: {metadata_str}")
            return "general", ["basic"]
            
        task = task_match.group(1).strip()
        skills = [s.strip() for s in skills_match.group(1).split(',') if s.strip()]
        
        if not skills:
            skills = ["basic"]
        
        return task, skills
    except Exception as e:
        logger.error(f"Failed to parse metadata: {metadata_str}")
        logger.error(f"Error details: {str(e)}")
        return "general", ["basic"]


def generate_rubrics_actions(use_case: str, skills: List[str], number_of_rubrics: int = 5) -> str:
    prompt = """I want you to act as a instruction judge with domain expertise.
Your job is to generate {number_of_rubrics} domain specific rubrics to assess the difficulty and complexity based on the use case of the instruction, and skills required to respond to it.
The generated rubrics should be clear, concise and unambiguous.
Based on the generated rubrics, generate corresponding actions to improve an instruction by making it more challenging.

The use case of the instruction: {use_case}
The skills required to solve the instruction: {skills}

Generate the domain-specific rubrics and actions without explanation in numbered bulletin points.

Output format:
Rubrics:
1. [rubric description]
...
{number_of_rubrics}. [rubric description]

Actions:
Action 1: [action description]
...
Action {number_of_rubrics}: [action description]

Output:
"""
    return prompt.format(use_case=use_case, skills=", ".join(skills), number_of_rubrics=number_of_rubrics)


def improve_instruction(instruction: str, action: str) -> str:
    prompt = """I want you to act as a instruction improver with domain expertise.
Your job is to make the given instruction more challenging following the given improving action item, and the generated instruction should be reasonable and self-consistent.
Do not directly copy words or phrases in the action.
Please generate exactly ONE improved version of the instruction and do not generate any other text.

### Example:
Improving action: Add error handling
Input instruction: Write a function to sort a list
Improved instruction: Implement a robust sorting function that handles invalid inputs like None values, non-numeric elements, and empty lists. The function should raise appropriate exceptions with descriptive error messages.

### Your Task:
Improving action: {action}
Input instruction: {instruction}
Improved instruction: """
    return prompt.format(action=action, instruction=instruction)


def select_random_action(actions_str: str) -> str:
    """Select a random action from generated rubrics."""
    try:
        if "Rubrics:" not in actions_str:
            logger.error(f"No 'Rubrics:' section found in:\n{actions_str}")
            return "make it more specific and challenging"
            
        rubrics_section = actions_str.split("Rubrics:")[1].strip()
        
        actions = []
        pattern = r'(?:^|\n)(?:\*\*)?(?:Action\s*)?(\d+[\)\.:]|\d+\.\s*\*\*)\s*(.+?)(?=(?:\n\d+[\)\.:]|\n\*\*|\n$|$))'
        matches = re.finditer(pattern, rubrics_section, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            # Extract the text, remove asterisks and clean up
            action_text = match.group(2)
            action_text = re.sub(r'\*\*', '', action_text)  # Remove asterisks
            action_text = re.sub(r'\s+', ' ', action_text)  # Normalize whitespace
            action_text = action_text.strip()
            
            # If action contains a colon, take everything after it
            if ':' in action_text:
                action_text = action_text.split(':', 1)[1].strip()
                
            if action_text:
                actions.append(action_text)
        
        if not actions:
            logger.error(f"No valid actions found in rubrics section:\n{rubrics_section}")
            return "make it more specific and challenging"
            
        return random.choice(actions)
    except Exception as e:
        logger.error(f"Failed to select action: {str(e)}\nInput text:\n{actions_str}")
        return "make it more specific and challenging"


get_response_template = """Below is an instruction that describes a task, write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


@dataclass
class Config:
    input_file: str = "/home/zhe/tag-instruct-experiment/magpie_5k.jsonl"
    output_file: str = "/home/zhe/tag-instruct-experiment/result/codeclm-qwen72b-magpie"    
    iter_num: int = 5
    max_tokens: int = 2048
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)


# def instruction_cleaning(instruction: str) -> str:
#     # Remove leading/trailing quotes
#     instruction = instruction.strip('"')
    
#     # Check for any numbered pattern like "1." or "1)"
#     if re.search(r'\d+[\.\)]', instruction):
#         # Extract first numbered item
#         match = re.search(r'\d+[\.\)]\s*(.+?)(?=\s*\d+[\.\)]|\s*$)', instruction)
#         if match:
#             return match.group(1).strip()
#     # Split by double newlines and take first part
#     parts = instruction.split('\n\n')
#     if len(parts) > 1:
#         return parts[1].strip().strip('"').strip()
#     return parts[0].strip().strip('"').strip()  # Return first part if there's no double newline


def process_instructions(config: Config):
    data = load_jsonlines(config.input_file)
    
    for i in range(config.iter_num):
        logger.info(f"Starting iteration {i + 1}")
        
        # Step 2: Extract metadata from instructions
        metadata_prompts = [extract_metadata(item['instruction']) for item in data]
        metadata_results = parallel_inference(
            metadata_prompts, 
            max_tokens=config.max_tokens,
            **vars(config.inference_config)
        )
        metadata_list = [parse_metadata(m) for m in metadata_results]
    
        
        rubrics_prompts = [generate_rubrics_actions(m[0], m[1]) for m in metadata_list]
        actions_list = parallel_inference(
            rubrics_prompts,
            max_tokens=config.max_tokens,
            **vars(config.inference_config)
        )
        
    
        original_instructions = [item['instruction'] for item in data]
        improve_prompts = [improve_instruction(instr, select_random_action(acts))
                          for instr, acts in zip(original_instructions, actions_list)]
        
        improved_instructions = parallel_inference(
            improve_prompts,
            max_tokens=config.max_tokens,
            **vars(config.inference_config)
        )
        
        responses = parallel_inference(
            [get_response_template.format(instruction=instr) for instr in improved_instructions],
            max_tokens=config.max_tokens,
            **vars(config.inference_config)
        )
        
        # Step 7: Prepare output data
        generated_data = [
            {
                "instruction": improved,
                "response": resp,
                "metadata": {"use_case": m[0], "skills": m[1]}
            }
            for m, improved, resp in zip(metadata_list, improved_instructions, responses)
        ]
        
        write_jsonlines(generated_data, f"{config.output_file}/codeclm_iter_{i}.jsonl")
        data = generated_data


if __name__ == "__main__":
    config = Config()
    process_instructions(config)
    
