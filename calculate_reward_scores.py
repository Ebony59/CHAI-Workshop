import os
import sys
import re
import numpy as np
import random
import torch
import wandb
import warnings
import logging

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
    MODEL = "mistralai/Mistral-Small-Instruct-2409"
    ELO = 1224

    eval_dataset = load_dataset('ChaiML/reward_formatted_blend_mokul_2024-11-14_100_convos', split='train')
    eval_dataset = eval_dataset.select_columns(['payload'])

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map='auto',
    )
    # model.to('cuda')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Initialize the reward model
    # reward_model_repo = "ChaiML/gpt2xl-classification-rm-edit-unfrozen-73acc"
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

    # Evaluate the model with the reward model
    model.eval()
    
    reward_scores = []
    alignment_scores = []
    generated_texts = []
    for i in tqdm(range(len(eval_dataset))):
        prompt = eval_dataset[i]['payload']
        text_generator = pipeline("text-generation", model=model.half(), tokenizer=tokenizer)
        generated_text = text_generator(
            prompt + '####\n',
            max_new_tokens=200,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = generated_text[0]['generated_text'].split('\n####\n')[1]
        generated_text = generated_text.split('\n')[0]
        generated_texts.append(generated_text)

        if i == 0:
            print('prompt:',prompt)
            print('generated_text:',generated_text)

        prompt = prompt.strip('\n')

        reward_input = f"{prompt} {generated_text}"
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

        reward_scores.append(reward_score)
        alignment_scores.append(alignment_score)

    reward_score = np.mean(reward_scores)
    reward_score_std = np.std(reward_scores)

    alignment_score = np.mean(alignment_scores)
    alignment_score_std = np.std(alignment_scores)

    print(f"Reward Score: {reward_score} Std: {reward_score_std}; Alignment Score: {alignment_score} Std: {alignment_score_std}")

    with open('./reward_scores.csv','a') as f:
        f.write(f'{MODEL},{ELO},{reward_score},{reward_score_std},{alignment_score},{alignment_score_std}\n')
