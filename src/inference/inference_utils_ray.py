from typing import List, Dict
import re
from loguru import logger
import ray
from vllm import LLM, SamplingParams
import logging



gpus = ["1" , "2" , "3" , "4", "5", "6", "7"]
num_gpus = len(gpus)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)



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




def get_template(prompt: str, template_type: str = "default", tokenizer = None) -> str:
    if template_type == "alpaca":
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""
    elif template_type == "mistral":
        return f"""<|im_start|>user
{prompt}
<|im_end|>
"""
    elif template_type == "direct":
        return prompt
    else:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)




@ray.remote(num_gpus=1)
def vllm_inference(model_name_or_path: str, input_data: List[str], max_tokens: int = 256, 
                  temperature: float = 0, top_p: float = 0.9, skip_special_tokens: bool = True, 
                  logical_gpu_id: int = 0, num_beams: int = None, template_type: str = "default"):
    
    config = get_model_config(model_name_or_path)
    llm = LLM(model=model_name_or_path, tokenizer_mode="auto", trust_remote_code=True, 
              max_model_len=config["max_model_len"], gpu_memory_utilization=0.95)
    
    # Apply template to input prompts
    tokenizer = llm.get_tokenizer()
    input_data = [get_template(prompt, template_type, tokenizer) for prompt in input_data]
    
    # Configure sampling parameters including beam search
    sampling_params_dict = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "skip_special_tokens": skip_special_tokens
    }
    
    if num_beams is not None and num_beams > 1:
        sampling_params_dict.update({
            "num_beams": num_beams,
            "best_of": num_beams,
            "use_beam_search": True
        })
    
    # Add model-specific stop tokens
    if "llama3" in model_name_or_path:
        sampling_params_dict["stop_token_ids"] = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    sampling_params = SamplingParams(**sampling_params_dict)
    outputs = llm.generate(input_data, sampling_params)
    return [output.outputs[0].text for output in outputs]

def parser_score(input_list: List[str]) -> List[int]:
    pattern = re.compile(r'score:\s*(\d)', re.IGNORECASE)
    scores = [int(match.group(1)) if (match := pattern.search(s)) else 0 for s in input_list]
    return scores


def parallel_inference(prompt_list: List[Dict], model_name_or_path: str, max_tokens: int = 256, 
                      temperature: float = 0, top_p: float = 0.9, skip_special_tokens: bool = True, 
                      score: bool = False, num_beams: int = None, template_type: str = "default"):
    ray.init(log_to_driver=False, logging_level=logging.WARNING, num_gpus=num_gpus)
    chunk_size = len(prompt_list) // num_gpus
    chunks = [prompt_list[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)]
    if len(prompt_list) % num_gpus != 0:
        chunks[-1].extend(prompt_list[num_gpus * chunk_size:])
    
    for i in range(num_gpus):
        logger.warning(f"chunk {i} size: {len(chunks[i])}")

    tasks = [vllm_inference.remote(model_name_or_path, chunk, max_tokens if not score else 10,
                                 temperature, top_p, skip_special_tokens, i, num_beams, template_type) 
             for i, chunk in enumerate(chunks)]
    
    processed_data = sum(ray.get(tasks), [])
    ray.shutdown()
    return processed_data if not score else parser_score(processed_data)


if __name__ == "__main__":
    model_name_or_path = "/home/admin/data/huggingface_model/LLaMA/Meta-Llama-3-8B-Instruct"
    prompt_list = ["Hello, how are you?", "What is the meaning of life?"]*100
    result = parallel_inference(prompt_list, model_name_or_path, max_tokens=256, temperature=0, top_p=0.95, skip_special_tokens=True)
    print(result)
    result = parallel_inference(prompt_list, model_name_or_path, max_tokens=256, temperature=0, top_p=0.95, skip_special_tokens=True, score=True)
    print(result)

    


