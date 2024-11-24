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

global reward_samples
reward_samples = 50

global layer
layer = random.choice(range(55))

global choisen_i
chosen_i = random.choice(range(50))
        

# Define the custom callback
class RewardLoggingCallback(TrainerCallback):
    def __init__(self, reward_model, alignment_model, tokenizer, reward_tokenizer, alignment_tokenizer, dataset, eval_steps=100):
        self.reward_model = reward_model
        self.alignment_model = alignment_model
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.alignment_tokenizer = alignment_tokenizer
        self.dataset = dataset
        self.eval_steps = eval_steps
        
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            with open('./lora_model_weights.txt','a') as f:
                f.write(f'step {state.global_step}\n')

            # Evaluate the model with the reward model
            reward_scores, alignment_scores = self.evaluate_with_reward_model(kwargs['model'])

            reward_score = np.mean(reward_scores)
            reward_score_std = np.std(reward_scores)

            alignment_score = np.mean(alignment_scores)
            alignment_score_std = np.std(alignment_scores)
            
            # Log the average reward to W&B
            wandb.log({"avg reward scores": reward_score, "step": state.global_step})
            wandb.log({"reward scores std": reward_score_std, "step": state.global_step})

            wandb.log({"avg alignment scores": alignment_score, "step": state.global_step})
            wandb.log({"alignment scores std": alignment_score_std, "step": state.global_step})
            print(f"Step {state.global_step}: Reward Score: {reward_score} Std: {reward_score_std}; Alignment Score: {alignment_score} Std: {alignment_score_std}")
    
    def evaluate_with_reward_model(self, model): 
        model.eval()
        
        reward_scores = []
        alignment_scores = []
        generated_texts = []
        for i in range(len(self.dataset)):
            prompt = self.dataset[i]['payload']
            text_generator = pipeline("text-generation", model=model.half(), tokenizer=self.tokenizer)
            generated_text = text_generator(
                prompt + '####\n',
                max_new_tokens=200,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_text = generated_text[0]['generated_text'].split('\n####\n')[1]
            generated_text = generated_text.split('\n')[0]
            generated_texts.append(generated_text)
    
            reward_input = f"{prompt}\n####\n{generated_text}"
            reward_inputs = self.reward_tokenizer(reward_input, return_tensors="pt", padding=True, truncation=True).to("cuda")
            alignment_inputs = self.reward_tokenizer(reward_input, return_tensors="pt", padding=True, truncation=True).to("cuda")
            alignment_inputs = self.alignment_tokenizer(reward_input, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
            with torch.no_grad():
                reward_outputs = reward_model(**reward_inputs)
                alignment_outputs = alignment_model(**alignment_inputs)
                reward_score = reward_outputs.logits[:, 1].item()
                alignment_score = alignment_outputs.logits[:, 1].item()
    
            reward_scores.append(reward_score)
            alignment_scores.append(alignment_score)

        print('model response:', generated_texts[chosen_i])
        
        return reward_scores, alignment_scores


if __name__=='__main__':
    BASE_MODEL = "mistralai/Mistral-Small-Instruct-2409"
    MODEL_NAME = "EZStorytellingEditsSFT_9users_1k"

    with open('./lora_model_weights.txt','w') as f:
        f.write(f'layer: {layer}\n')

    # Initialize W&B
    wandb.init(project=MODEL_NAME)
    
    # Load dataset
    train_dataset = load_dataset('ChaiML/EZ_9users_edit_storytelling_5k_1000sample', split='train')
    train_dataset = train_dataset.select_columns(['text'])
    print('Length of dataset:', len(train_dataset))
    print('steps per epoch:', int(len(train_dataset)/16))

    # evaluate reward and alignment scores every half an epoch
    reward_eval_steps = int( (len(train_dataset)/16) / 2)

    # load eval dataset for reward model
    eval_dataset = load_dataset('ChaiML/reward_formatted_blend_mokul_2024-11-14_100_convos', split='train')
    eval_dataset = eval_dataset.select_columns(['payload'])

    eval_df = eval_dataset.to_pandas()
    eval_df = eval_df.sample(n=reward_samples).reset_index(drop=True)
    eval_dataset = Dataset.from_pandas(eval_df)

    print(f'payload for {chosen_i}th reward dataset')
    print(eval_dataset[chosen_i]['payload'])
    
    
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map='auto',
    )
    # model.to('cuda')
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    
    # Define data collator
    response_template = "####\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    # Load LoRA model
    lora_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.05,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    
    # Define training arguments
    training_args = TrainingArguments(
        num_train_epochs=4,
        learning_rate=1e-05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        do_eval=True,
        per_device_eval_batch_size=1,
        adam_epsilon=1e-08,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.1,
        seed=42,
        logging_steps=5,
        save_steps=1,
        eval_steps=20,
        save_strategy="epoch",
        output_dir=f"data/{MODEL_NAME}",
        hub_model_id="dpo",
        gradient_checkpointing=True,
        bf16=True,
        report_to=['wandb'],
    )
    
    # Initialize the reward model
    reward_model_repo = "ChaiML/gpt2xl-classification-rm-edit-unfrozen-73acc"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_repo).to("cuda")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_repo)

    # Initialize the alignment reward model
    alignment_model_repo = 'ChaiML/CHAI_alignment_reward_model'
    alignment_model = AutoModelForSequenceClassification.from_pretrained(reward_model_repo).to("cuda")
    alignment_tokenizer = AutoTokenizer.from_pretrained(alignment_model_repo)
    
    print("Reward models and tokenizers loaded successfully!")
    
    # Initialize the custom callback
    reward_logging_callback = RewardLoggingCallback(
        reward_model=reward_model,
        alignment_model=alignment_model,
        tokenizer=tokenizer,
        reward_tokenizer=reward_tokenizer,
        alignment_tokenizer=alignment_tokenizer,
        dataset=eval_dataset,
        eval_steps=reward_eval_steps,  # Evaluate every 10 steps
    )
    
    # Initialize the trainer with the custom callback
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=1600,
        dataset_text_field="text",
        peft_config=lora_config,
        callbacks=[reward_logging_callback],  # Add the custom callback here
    )
    
    # Train the model
    trainer.train()
    trainer.save_model()
    
    # Push to Hub
    trained_model = model.merge_and_unload()
    tokenizer.push_to_hub(f'ChaiML/{MODEL_NAME}', private=True)
    trained_model.push_to_hub(f'ChaiML/{MODEL_NAME}', private=True)
    
    # Trained model verification
    text_generator = pipeline("text-generation", model=trained_model.half(), tokenizer=tokenizer)
    prompt, expected_response = train_dataset['text'][0].split('\n####\n')
    generated_text = text_generator(
        prompt + '\n####\n',
        max_new_tokens=200,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = generated_text[0]['generated_text'].split('\n####\n')[1]
    print(f'Expected response: {expected_response}')
    print(f'Generated response: {generated_text}')

    formatter = PygmalionFormatter()
    formatter.memory_template = ''
    formatter.prompt_template = ''

    generation_params={
        'frequency_penalty': 0.5,
        'max_input_tokens': 1024,
        'presence_penalty': 0.5,
        'stopping_words': ['\n'],
        'temperature': 0.9,
        'top_k': 80,
        'top_p': 0.95,
        'min_p': 0.05,
        'best_of': 4,
    }

    submission_parameters = {
        "model_repo": f"ChaiML/{MODEL_NAME}",
        "generation_params": generation_params,
        # "formatter": formatter,
    }
    
    # Submit the model
    submitter = ModelSubmitter(verbose=True)
    submitter.submit(submission_parameters)
