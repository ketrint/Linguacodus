# -*- coding: utf-8 -*-
"""
Script Name: llamafinetune.py
Purpose: Llama 2 fine-tune for code instruction generation for a given ML task
"""

import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from sklearn.utils import shuffle
from datasets import Dataset


# Configuration variables
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
NEW_MODEL_PATH = "./models/llama-2-7b-supercode"
CSV_FILEPATH = '/data/input_prompts.csv'
OUTPUT_DIR = "./models"
DEVICE_MAP = {"": 0}
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "optim": "paged_adamw_32bit",
    "save_steps": 500,
    "logging_steps": 25,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": False,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "constant",
    "report_to": "tensorboard"
}


def load_data(filepath):
    """
    Load and shuffle data from a CSV file.
    """
    df = pd.read_csv(filepath)
    df = shuffle(df)
    return Dataset.from_pandas(df)
    
    
def load_model_and_tokenizer():
    """
    Load the model and tokenizer with QLoRA configuration.
    """
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and USE_4BIT:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=DEVICE_MAP
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
    
    
def create_trainer(model, tokenizer, dataset):
    """
    Create the SFTTrainer for supervised fine-tuning.
    """
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(**TRAINING_ARGS)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="prompt",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,  # Adjust this if you want to enable packing
    )

    return trainer


def main():
    """
    Main function to run the script.
    """
    try:
        dataset = load_data(CSV_FILEPATH)
        model, tokenizer = load_model_and_tokenizer()
        trainer = create_trainer(model, tokenizer, dataset)
        
        trainer.train()
        trainer.save_model(NEW_MODEL_PATH)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
