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
    
    
    
# Initial evolving method template
best_instruction = """You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.
Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

Step 1: Please read the "#Instruction#" carefully and list all the possible methods to make this instruction more complex (to
make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to
change the language of the instruction!

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more
complex. The plan should include several methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to
20 words into the "#Instruction#".

Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten
Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# without any
explanation.

Please reply strictly in the following format:
Step 1 #Methods List#:
Step 2 #Plan#:
Step 3 #Rewritten Instruction#:
Step 4 #Finally Rewritten Instruction#:

#Instruction#:
{instruction}"""

get_response_template = """Below is an instruction that describes a task, write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
def parse_instruction(instruction: str, original_instruction: str):
    # List of possible markers to look for
    markers = [
        "#Finally Rewritten Instruction#:",
        "#Finally Rewritten Instruction#",
        "Step 4:",
        "#Rewritten Instruction#:"
    ]
    
    # Try each marker
    for marker in markers:
        last_index = instruction.lower().rfind(marker.lower())
        if last_index != -1:
            # Get everything after the marker
            final_instruction = instruction[last_index + len(marker):].strip()
            
            # Remove any quotes at start/end and extra whitespace
            final_instruction = final_instruction.strip('"').strip("'").strip()
            
            # Validate the instruction is not empty
            if final_instruction:
                return final_instruction
    
    # If no valid instruction was found, log the error and return original instruction
    logger.error(f"Could not parse instruction from response: {instruction}")
    return original_instruction

@dataclass
class Config:
    input_file: str = "/home/zhe/tag-instruct-experiment/magpie_5k.jsonl"
    output_dir: str = "/home/zhe/tag-instruct-experiment/result/autoevol-qwen72b-magpie"
    iter_num: int = 5
    max_tokens: int = 2048
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)

def process_instructions(config: Config):
    # Load initial data
    data = load_jsonlines(config.input_file)

    # Iterate multiple times
    for i in range(config.iter_num):
        output_file = f"{config.output_dir}/autoevol_{i}.jsonl"
        if os.path.exists(output_file):
            logger.info(f"Skipping iteration {i} as it already exists")
            data = load_jsonlines(output_file)
            continue
            
        logger.info(f"Starting iteration {i + 1}")
        
        # Generate new instructions
        prompts = [best_instruction.format(instruction=item['instruction']) for item in data]
        generated_instructions = parallel_inference(prompts, max_tokens=config.max_tokens, **vars(config.inference_config))
        
        # Parse the generated instructions
        parsed_instructions = [parse_instruction(gen_inst, item['instruction']) 
                             for gen_inst, item in zip(generated_instructions, data)]
        logger.info(f"Parsed instructions: {parsed_instructions}")  
        
        # Generate responses for new instructions 
        prompts = [get_response_template.format(instruction=item) for item in parsed_instructions]
        responses = parallel_inference(prompts, max_tokens=config.max_tokens, **vars(config.inference_config))
        
        # Prepare output data - ensure we only iterate over the available items
        new_data = [
            {"instruction": instruction.strip('"').strip(),
             "response": response} 
            for instruction, response in zip(parsed_instructions, responses)
        ]
        
        # Save current iteration and update data for next iteration
        os.makedirs(config.output_dir, exist_ok=True)
        write_jsonlines(new_data, output_file)
        data = deepcopy(new_data)  # Use current generation as seed for next iteration
        
        logger.info(f"Iteration {i + 1} completed. Data saved to {output_file}")

if __name__ == "__main__":
    config = Config()
    process_instructions(config) 
    
