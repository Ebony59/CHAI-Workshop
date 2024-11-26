import os
import sys
import re
import numpy as np
import random
import torch
import wandb
import warnings
import logging
import pandas as pd

from datasets import *
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    pipeline,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from chaiverse.submit import ModelSubmitter
from chaiverse.formatters import PygmalionFormatter

if __name__=='__main__':
    cut = 0
    DATA_REPO = 'ChaiML/EZ_12users_edit_storytelling'
    
    dataset = load_dataset(DATA_REPO)
    df = dataset['train'].to_pandas()

    reward_model_repo = "ChaiML/gpt2_xl_pairwise_89m_step_347634"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_repo).to("cuda")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_repo)

    if not reward_tokenizer.pad_token:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    # Initialize the alignment reward model
    alignment_model_repo = 'ChaiML/CHAI_alignment_reward_model'
    alignment_model = AutoModelForSequenceClassification.from_pretrained(alignment_model_repo).to("cuda")
    alignment_tokenizer = AutoTokenizer.from_pretrained(alignment_model_repo)

    print("Reward models and tokenizers loaded successfully!")

    df['reward_score'] = ''
    df['alignment_score'] = ''

    for i in tqdm(range(len(df))):
        text = df.loc[i,'text']
        bot_name = df.loc[i,'bot_name']
        prompt, response = text.split("\n####\n")
        reward_input = f"{prompt} {response}"
        
        reward_inputs = reward_tokenizer(reward_input, return_tensors="pt", padding=True, truncation=True).to("cuda")
        alignment_inputs = reward_tokenizer(reward_input, return_tensors="pt", padding=True, truncation=True).to("cuda")
        alignment_inputs = alignment_tokenizer(reward_input, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            reward_outputs = reward_model(**reward_inputs)
            alignment_outputs = alignment_model(**alignment_inputs)

            if reward_outputs.logits.shape == torch.Size([1, 1]):
                reward_score = reward_outputs.logits.item()
            else:
                reward_score = reward_outputs.logits[:, 1].item()
            if alignment_outputs.logits.shape == torch.Size([1, 1]):
                alignment_score = alignment_outputs.logits.item()
            else:
                alignment_score = alignment_outputs.logits[:, 1].item()


        df.loc[i,'reward_score'] = reward_score
        df.loc[i,'alignment_score'] = alignment_score

    df = df[df['reward_score'] > cut].reset_index(drop=True)

    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(f'{DATA_REPO}_reward', private=True)
