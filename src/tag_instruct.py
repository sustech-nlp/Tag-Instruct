import random
random.seed(42)
from inference.vllm_client import parallel_inference
from utils import *
from loguru import logger
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os

# Template for generating tags or categories
@dataclass
class TagInstructConfig:
    input_file: str = "/home/zhe/tag-instruct-experiment/alpaca_5k.jsonl" # use magpie 
    output_dir: str = "/home/zhe/tag-instruct-experiment/result/tag-instruct-thinkdifferent" # generate tag-instruct magpie
    iter_num: int = 5
    max_tokens: int = 2048
    inference_config: InferenceConfig = InferenceConfig()
    

get_response_template = '''Below is an instruction that describes a task, write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''


get_tags_template = '''You are a semantic analysis expert. Your task is to extract exactly three essential tags that capture the core semantic concepts of the given instruction or question. The tags should be:
- Concise (1-2 words each)
- Hierarchically ordered by importance
- Generalizable across similar tasks

### Guidelines:
1. Focus on action-oriented concepts
2. Avoid redundant or overlapping tags
3. Use standard terminology when possible

### Examples:

[Input]
Describe a situation where team collaboration improved the outcome of a project.

[Tags]
teamwork_experience, project_outcomes, success_factors

[Input]
What strategies can be used to improve time management in a busy work environment?

[Tags]
productivity_methods, workload_optimization, efficiency_tactics

[Input]
Outline the steps required to create a successful marketing campaign for a new product launch.

[Tags]
campaign_planning, market_strategy, launch_execution

[Input]
In the context of climate change adaptation, analyze how urban planning strategies can be modified to create resilient cities that can withstand extreme weather events while maintaining economic growth and social equity for their residents over the next 50 years.

[Tags]
urban resilience, climate adaptation, sustainable development

[Input]
Design a comprehensive employee training program for a multinational corporation that addresses cultural sensitivity, remote work effectiveness, and digital tool proficiency while ensuring consistent skill development across different time zones and accounting for various learning styles and language barriers.

[Tags]
corporate training, global workforce, skill development

[Input]
Develop a detailed analysis of how artificial intelligence implementations in healthcare systems can improve patient outcomes while considering privacy concerns, medical ethics, and the integration challenges with existing hospital infrastructure and staff training requirements.

[Tags]
healthcare AI, medical ethics, system integration

### Task:
Given the following instruction, provide exactly three semantic tags following the above format and guidelines:

[Input]
{instruction}

[Output Format]
tag1, tag2, tag3

[Tags]
'''


def add_tags_only_template(tags, question):
    prompt = '''Please generate only one new tag based on the existing tags and the provided question.
This new tag should be relevant to the question but introduce a unique perspective, add meaningful depth, and make the question more challenging.
Start by briefly explaining why you chose this new tag, then provide the tag itself.

### Format:
[Tags]
List of existing tags.

[Original Question]
The original question content.

Output:
[Explanation]
Explanation of the new tag.

[New Tag]
The new tag.

### Your Task:
[Tags]
{tags}

[Original Question]
{question}

Output:
'''
    return prompt.format(tags=tags, question=question)

def add_tags_only_template_think_different(tags, question, num_tags=5):
    prompt = f'''As an AI expert in semantic analysis, your task is to:
1. First, thoughtfully analyze the existing tags and question
2. Then, brainstorm and generate {num_tags} diverse, high-quality tags that could add depth to the topic
3. Finally, select the single most impactful tag to enhance the question, and strictly follow the format.

[Explanation]
Your detailed explanation here...

[New Tag]
your_chosen_tag


### Step 1: Analysis
First, think about:
- The core concepts in the existing tags and question
- Potential unexplored angles and perspectives
- Different domains this topic could connect to
- Various levels of abstraction (technical, theoretical, practical, etc.)

### Step 2: Generate {num_tags} Potential Tags
Generate {num_tags} unique tags, each with a brief explanation of its value:

[Tag List]
1. tag_name: explanation of why this tag adds value
2. tag_name: explanation of why this tag adds value
...continue until {num_tags} tags

### Step 3: Final Selection
After considering all options, select the single most valuable new tag that:
- Introduces a unique perspective
- Adds meaningful depth
- Makes the question more challenging
- Complements existing tags without redundancy

### Format:
[Existing Tags]
{tags}

[Original Question]
{question}

Your response must follow this exact format:
1. First write "### Step 1: Analysis" and provide your analysis
2. Then write "### Step 2: Generate {num_tags} Potential Tags" and list your tags with explanations
3. Finally, write:
[Explanation]
Your detailed explanation here...

[New Tag]
your_chosen_tag

Do not add any other text or sections. Strictly follow this output format.
'''
    return prompt



def get_instructions_from_tags(tags, question):
    prompt = '''Using the provided tags and original question, generate a new version of the question that is more complex and thought-provoking.

### Requirements:
1. **Tag Selection**: Choose a subset of the tags that best enhance the depth of the question.
2. **New Question Complexity**: The new question should be more challenging than the original, requiring deeper thought, but it should be solvable. Avoid simply adding length; instead, focus on making the question more insightful and intellectually engaging.
3. **Solvability**: Ensure the new question remains clear and achievable.

### Output Format:
[Tags]
Here are the existing tags.

[Original Question]
Here is the original question.

Output:
[New Question]
Provide the new, more complex question.


You only need to return the new question; no other information: 
### Your Task
[Tags]
{tags}

[Original Question]
{question}

Output:
[New Question]
'''
    return prompt.format(tags=tags, question=question)


def clean_instruction(instruction: str) -> str:
    # First try to extract content after [New Question] tag
    if "[New Question]" in instruction:
        instruction = instruction.split("[New Question]")[1].strip()
    
    # # Remove any remaining markdown-style tags
    # instruction = re.sub(r'\[.*?\]', '', instruction)
    
    # Clean up any extra whitespace
    instruction = instruction.strip()
    
    return instruction
    
    
inference_config = InferenceConfig()

#  Operator: Addition;
def parsing_explan_tag(prompt: str) -> Tuple[str, str]: 
    explan_parttern = r'\[Explanation\]\s*(.*)\s*\[New Tag\]'
    new_tag_pattern = r'\[New Tag\]\s*(.*)'
    new_tag = re.search(new_tag_pattern, prompt, re.DOTALL)
    new_explan = re.search(explan_parttern, prompt, re.DOTALL)
    
    return [
        new_explan.group(1).strip() if new_explan else None, 
        new_tag.group(1).strip() if new_tag else None
    ]
    

def parsing_only_question(prompt: str) -> str:
    new_question_pattern = r'\[New Question\]\s*(.*)'
    new_question = re.search(new_question_pattern, prompt, re.DOTALL)
    return new_question.group(1).strip() if new_question else None



def get_responses(data: List[dict]) -> List[dict]:
    instructions = [entry["instruction"] for entry in data]
    prompts = [get_response_template.format(instruction=instruction) for instruction in instructions]
    responses = parallel_inference(prompts, max_tokens=2048, **vars(inference_config))
    for i, response in enumerate(responses):
        data[i]["response"] = response
    return data
 
 
# Operator 1: Addition
@save_or_skip_dynamic('file_pth')
def ADDITION(data: List[dict], **kwargs) -> List[dict]:
    
    data = get_tags(data)
    instructions = [entry["instruction"] for entry in data]
    tags = [entry["tag"] for entry in data]
    
    # ADDITION Tag:
    prompts = [add_tags_only_template(tags=tag, question=instruction) for instruction, tag in zip(instructions, tags) if tag and instruction]
    outputs = parallel_inference(prompts, max_tokens=2048, **vars(inference_config))
    # print(outputs)
    results = [parsing_explan_tag(output) for output in outputs]
    new_tags = [result[1].strip() for result in results if result[1]]
    all_tags = [f"{tag}, {add_tag}" for tag, add_tag in zip(tags, new_tags)]
    
    
    # Generate prompts for creating new questions
    prompts = [get_instructions_from_tags(tags=tag, question=instruction) for instruction, tag in zip(instructions, all_tags)]
    outputs = parallel_inference(prompts, max_tokens=1024, **vars(inference_config))
    results = [clean_instruction(output) for output in outputs]
    new_generated_data = [{"instruction": result, "tag": tag} for result, tag in zip(results, all_tags) if result[0] and result[1]]
    new_generated_data = get_responses(new_generated_data)
    return new_generated_data

def get_tags(data: List[dict]) -> List[dict]:
    instructions = [entry["instruction"] for entry in data]
    prompts = [get_tags_template.format(instruction=instruction) for instruction in instructions]
    outputs = parallel_inference(prompts, max_tokens=256, **vars(inference_config))
    for i, output in enumerate(outputs):
        data[i]["tag"] = output.strip()
    return data

    
def process_tag_instructions(config: TagInstructConfig):
    """Main processing function for tag-based instruction generation"""
    # Load initial data
    data = get_seed_data(config.input_file)
    
    # Iterate multiple times
    for i in range(config.iter_num):
        output_file = f"{config.output_dir}/tag_instruct_{i}.jsonl"
        if os.path.exists(output_file):
            logger.info(f"Skipping iteration {i} as it already exists")
            data = load_jsonlines(output_file)  
            continue
            
        logger.info(f"Starting iteration {i + 1}")
        
        # Generate new complex instructions with additional tags
        complex_new_data = ADDITION(data, file_pth=output_file)
        data = complex_new_data  # Use current generation as seed for next iteration
        
        logger.info(f"Iteration {i + 1} completed. Data saved to {output_file}")



if __name__ == "__main__":
    config = TagInstructConfig()
    process_tag_instructions(config)




