from loguru import logger
from dataclasses import dataclass, field
from inference.vllm_client import parallel_inference
from utils import load_jsonlines, write_jsonlines
import os
from copy import deepcopy
import random
random.seed(42)


@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    model_name_or_path: str = "/home/zhe/.cache/modelscope/hub/qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
    
    
    
    

# 每次增加3个节点, 循环5次 
base_instruction = """You are an instruction rewriter. You need to rewrite a given user instruction following Procedure and all the Requirements. You MUST ONLY return the NEW instruction you rewrite.

Procedure:
step-1: Parse the old "instruction" to a TREE-1 through semantic parsing in natural language processing.
step-2: EXPAND the above NEW TREE-1 from DEPTH or WIDTH by ADDING Three meaningful NEW NODEs as Nouns or Verbs to form a NEW TREE-2. The new nodes should be constructed with detailed and pertinent information.
step-3: Generate a totally NEW "instruction" based on the expanded NEW TREE-2. 

Requirements:
1. Generate ONLY ONE NEW "INSTRUCTION" based on old "instruction" following "Procedure" in at most 100 words. 
2. Make sure the new "instruction" is reasonable, clear, more COMPLEX and at least 12 words longer than the old "instruction". 
2.1 Do not generate new nodes more than required.
3. Maintain the DATA FORMAT and the LANGUAGE of the old instruction. 
4. NOT return any sentences or steps in the "Procedure" or "Requirements": e.g., "adding three", "tree-1,2", "semantic parsing" are NOT allowed.
5. NO details or explanations for the instruction generation process.
6. NO response to the old or new instructions.
7. NO URLs or websites appears unless the old instruction has them.

Old instruction: "{instruction}"

New instruction: """

# response_template = """You are an Assistant, you need to generate a response for a given instruction by a User:

# You can refer to the following example:
# </example>
# example-Instruction: Explain why 4/16 is equivalent to 1/4.
# example-Response: The fraction 4/16 is equivalent to 1/4 because both fractions represent the same value. A fraction can be simplified by dividing both the numerator and the denominator by a common factor. In this case, 4 is a common factor of both the numerator and denominator of 4/16. When we divide both by 4, we get 4/4 = 1 and 16/4 = 4, so the simplified fraction is 1/4. Alternatively, we can think of this in terms of multiplication. For example, if we multiply the numerator and denominator of the fraction 1/4 by 4, we get (1x4)/(4x4), or 4/16. Since both fractions can be derived from the other through multiplication or division by the same number, they represent the same value and are equivalent.
# </example>

# You MUST:
# - Carefully understand the following 'Instruction' that describes a task. 
# - Provide a pricise response NO MORE THAN 1000 words to complete the task described in the 'Instruction'. 
# - If the task involves complexity and requires reasoning, you can break it down to sub-tasks and complete them step-by-step. 
# - Make sure the response is concise, accurate, and FULLY meets the requirements outlined in the 'Instruction'.
# - If your response is short, also give the explanation.
# - NO URL or websites in the response, unless the instruction asks for it.

# Instruction: "{instruction}"

# Response: """


get_response_template = """Below is an instruction that describes a task, write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

@dataclass
class Config:
    input_file: str = "/home/zhe/tag-instruct-experiment/magpie_5k.jsonl"
    output_dir: str = "/home/zhe/tag-instruct-experiment/result/tree-instruct-qwen72b-magpie"
    iter_num: int = 5
    max_tokens: int = 2048
    inference_config: InferenceConfig = InferenceConfig()

def process_instructions(config: Config):
    # Load initial data
    data = load_jsonlines(config.input_file)

    # Iterate multiple times
    for i in range(config.iter_num):
        output_file = f"{config.output_dir}/tree_instruct_{i}.jsonl"
        if os.path.exists(output_file):
            logger.info(f"Skipping iteration {i} as it already exists")
            continue
            
        logger.info(f"Starting iteration {i + 1}")
        
        # Generate new instructions
        prompts = [base_instruction.format(instruction=item['instruction']) for item in data]
        generated_instructions = parallel_inference(prompts, max_tokens=config.max_tokens, **vars(config.inference_config))
        
        # Generate responses for new instructions 
        prompts = [get_response_template.format(instruction=item) for item in generated_instructions]
        responses = parallel_inference(prompts, max_tokens=config.max_tokens,  **vars(config.inference_config))
        
        # Prepare output data
        new_data = [
            {"instruction": generated_instructions[i].strip('"').strip(),
             "response": responses[i]} 
            for i, item in enumerate(data)
        ]
        
        # Save current iteration and update data for next iteration
        os.makedirs(config.output_dir, exist_ok=True)
        # save_jsonlines(new_data, output_file)
        write_jsonlines(new_data, output_file)
        data = deepcopy(new_data)  # Use current generation as seed for next iteration
        
        logger.info(f"Iteration {i + 1} completed. Data saved to {output_file}")

if __name__ == "__main__":
    config = Config()
    process_instructions(config)
    