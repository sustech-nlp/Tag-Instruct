import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from loguru import logger

from inference.vllm_client import parallel_inference
from utils import load_json, load_jsonlines, write_jsonlines
random.seed(42)
from utils import instruction_cleaning



@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    model_name_or_path: str = "/home/zhe/.cache/modelscope/hub/qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
    
    
    

base_instruction = """I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#.
You SHOULD complicate the given prompt using the following method:
{}
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.
'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#"""

def createConstraintsPrompt(instruction):
    prompt = base_instruction.format("Please add one more constraints/requirements into #The Given Prompt#'")
    prompt += "\n#The Given Prompt#:\n{}\n".format(instruction)
    prompt += "#Rewritten Prompt#:\n"
    return prompt

def createDeepenPrompt(instruction):
    prompt = base_instruction.format("If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
    prompt += "\n#The Given Prompt#:\n{}\n".format(instruction)
    prompt += "#Rewritten Prompt#:\n"
    return prompt

def createConcretizingPrompt(instruction):
    prompt = base_instruction.format("Please replace general concepts with more specific concepts.")
    prompt += "\n#The Given Prompt#:\n{}\n".format(instruction)
    prompt += "#Rewritten Prompt#:\n"
    return prompt

def createReasoningPrompt(instruction):
    prompt = base_instruction.format("If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
    prompt += "\n#The Given Prompt#:\n{}\n".format(instruction)
    prompt += "#Rewritten Prompt#:\n"
    return prompt

# base_instruction = """I want you act as a Prompt Creator.
# Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
# This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
# The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.
# The #Created Prompt# must be reasonable and must be understood and responded by humans.
# '#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#"""

# def createBreadthPrompt(instruction):
#     prompt = base_instruction
#     prompt += "\n#Given Prompt#:\n{}\n".format(instruction)
#     prompt += "#Created Prompt#:\n"
#     return prompt


@dataclass
class Config:
    seed_pth: str = "/home/zhe/tag-instruct-experiment/magpie_5k.jsonl" 
    output_dir: str = "/home/zhe/tag-instruct-experiment/result/evol-qwen72b-magpie"
    iter_num: int = 5
    max_tokens: int = 2048
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)


config = Config()

get_response_template = """Below is an instruction that describes a task, write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""



def evolve_instructions(config: Config):
    seed_data = load_json(config.seed_pth)
    
    for i in range(config.iter_num):
        if os.path.exists(f"{config.output_dir}/evol_{i}.jsonl"):
            logger.info(f"Skipping iteration {i} as it already exists")
            continue
        
        logger.info(f"Starting iteration {i + 1}")
        
        prompts = []
        for cur_obj in seed_data:
            instruction = cur_obj['instruction'].strip()
            evol_prompts = [
                createConstraintsPrompt(instruction),
                createDeepenPrompt(instruction),
                createConcretizingPrompt(instruction),
                createReasoningPrompt(instruction),
                # createBreadthPrompt(instruction)
            ]
            selected_evol_prompt = random.choice(evol_prompts)
            prompts.append(selected_evol_prompt)

        new_instructions = parallel_inference(prompts, max_tokens=config.max_tokens, **vars(config.inference_config))

        answers = parallel_inference(
            [get_response_template.format(instruction=prompt) for prompt in new_instructions],
            max_tokens=config.max_tokens,
            **vars(config.inference_config)
        )

        pair = [{"instruction": i, "response": a} for i, a in zip(new_instructions, answers)]
        seed_data = deepcopy(pair)
        write_jsonlines(seed_data, f"{config.output_dir}/evol_{i}.jsonl")
        logger.info(f"Iteration {i + 1} completed. Data saved to evol_{i}.jsonl")


if __name__ == "__main__":
    evolve_instructions(config)

