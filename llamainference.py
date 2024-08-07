# -*- coding: utf-8 -*-
"""
Script Name: llamainference.py
Purpose: Llama 2 inference for code instruction generation for a given ML task
"""

import pandas as pd
import tqdm
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
from peft import LoraConfig, PeftModel

# Configuration variables
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
NEW_MODEL_PATH = "./models/llama-2-7b-supercode"
CSV_FILEPATH = './data/inputs.csv'
OUTPUT_FILEPATH = './data/results.csv'
PIPELINE_CONFIG = {
    "task": "text-generation",
    "max_length": 1024,
    "temperature": 0.7,
    "return_full_text": False
}

def load_model(model_name, new_model_path):
    """
    Load the pre-trained model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, new_model_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer
    

def load_data(filepath):
    """
    Load input data from a CSV file.
    """
    return pd.read_csv(filepath)
    

def generate_instructions(row, pipe):
    """
    Generate instructions for a given row of data.
    """
    task = row['task']
    data = row['data_type']
    metric = row['metrics']
    options = {}

    for place, option in zip(['first', 'second', 'third'], ['option_1', 'option_2', 'option_3']):
        prompt = f'''[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
        <</SYS>>
        Imagine that you are a data analyst. Your objective is writing the {place} place instruction for solving this machine learning task. Task: {task}
        The {data} data is used for the problem. The metric type is {metric} for the problem. Your response contains the main information about data preprocessing, model architecture and model training.''[/INST]'''

        generated_text = pipe(prompt)[0]['generated_text']
        options[option] = generated_text

    return {
        'task': task,
        'data_type': data,
        'metric': metric,
        'option_1': options['option_1'],
        'option_2': options['option_2'],
        'option_3': options['option_3'],
        'data_card': row['data_card'],
        'submission': row['submission'],
        'link': row['link']
    }
    

def save_results(results, output_filepath):
    """
    Save the results to a CSV file.
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filepath, index=False)


def main():
    """
    Main function to run the script.
    """
    try:
        model, tokenizer = load_model(MODEL_NAME, NEW_MODEL_PATH)
        data = load_data(CSV_FILEPATH)
        pipe = pipeline(model=model, tokenizer=tokenizer, **PIPELINE_CONFIG)

        results = []
        for _, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):
            result = generate_instructions(row, pipe)
            results.append(result)

        save_results(results, OUTPUT_FILEPATH)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
