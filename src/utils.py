import json
import os
from loguru import logger
from dataclasses import dataclass
from typing import List, Dict, Any
import re
from functools import wraps
import glob


def load_jsonlines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonlines(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        logger.info(f"Skipping operation because {file_path} already exists.")
        return
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

def load_json(file_path):
    if file_path.endswith(".jsonl"):
        return load_jsonlines(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        

def instruction_cleaning(texts: List[str]) -> List[str]:
    texts = [re.sub(r'\n+', '\n', s) for s in texts]
    texts = [re.sub(r"\d+\. ", "", text) for text in texts]
    texts = [re.sub(r'#', '', text) for text in texts] 
    def process_text(text: str) -> str:
        if '\n' in text:
            lines = text.split('\n')
            for line in lines:
                if len(line) > 10:
                    return line.strip()
        else:
            return text.strip()
    texts = [process_text(text) for text in texts]
    return texts



def get_seed_data(data_pth: str, type: str = "general") -> List[dict]:
    data = load_json(data_pth)
    if type== "Math":
        # question, answer
        data = [{"instruction": entry["question"], "response": entry["answer"]} for entry in data]
    elif type=="lima":
        data = [{"instruction": entry["instruction"] + "\n" + entry.get("input", "") if entry.get("input", "") == "" else entry["instruction"],
                "response": entry["output"]} for entry in data]
    else:
        data = [{"instruction": entry["instruction"], "response": entry["response"]} for entry in data]
    return data





def save_or_skip(file_pth_func):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 每次执行时获取最新的文件路径
            file_pth = file_pth_func()

            logger.info(f"Executing function: {func.__name__}")

            # 如果文件已经存在，加载并返回文件内容
            if os.path.exists(file_pth):
                logger.info(f"Skipping operation because {file_pth} already exists.")
                return load_jsonlines(file_pth)

            # 执行函数，获取结果
            result = func(*args, **kwargs)

            # 创建目录并保存结果到文件
            os.makedirs(os.path.dirname(file_pth), exist_ok=True)
            write_jsonlines(result, file_pth)

            # 统计文件夹中的文件数目
            num_files = len(glob.glob(os.path.join(os.path.dirname(file_pth), '*')))
            logger.info(f"Number of files in the directory: {num_files}")

            return result
        return wrapper
    return decorator


def calculate_avg_word_length(data):
    """统计每条 instruction 的单词长度，并返回平均值"""
    total_length = 0
    total_instructions = len(data)
    
    for item in data:
        instruction = item.get("instruction", "")
        word_count = len(instruction.split())  # 统计单词数
        total_length += word_count

    avg_length = total_length / total_instructions if total_instructions > 0 else 0    
    return avg_length



def save_or_skip_dynamic(parameter_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Executing function: {func.__name__}")

            # 获取 file_path 参数
            file_pth = kwargs.get(parameter_name)
            if not file_pth:
                raise ValueError(f"Parameter '{parameter_name}' not provided in kwargs.")
            
            if os.path.exists(file_pth):
                logger.info(f"Skipping operation because {file_pth} already exists.")
                data = load_jsonlines(file_pth)
            else:
                result = func(*args, **kwargs)
                os.makedirs(os.path.dirname(file_pth), exist_ok=True)
                write_jsonlines(result, file_pth)
                data = result

            # 统计单词长度，并计算平均值
            avg_length = calculate_avg_word_length(data)
            logger.warning(f"Number of results: {len(data)}")
            logger.warning(f"Average word count for instructions in {file_pth}: {avg_length}")

            # 记录目录中的文件数量
            num_files = len(glob.glob(os.path.join(os.path.dirname(file_pth), '*')))
            logger.info(f"Number of files in the directory: {num_files}")

            return data
        return wrapper
    return decorator








@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    model_name_or_path: str = "/data/zhe/mistralai--Ministral-8B-Instruct-2410"
    
    
    