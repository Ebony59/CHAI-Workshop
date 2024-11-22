import os
import re
import numpy as np
import random
import torch
import wandb
from datasets import load_dataset
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

global layer
layer = random.choice(range(55))


def evaluate_with_reward_model(model, tokenizer, reward_model, reward_tokenizer):   
    for name, param in model.named_parameters():
        if 'lora' in name:
            print(f"{name}: requires_grad={param.requires_grad}")
        
    for name, param in model.named_parameters():
        if ('norm' in name) or name=='lm_head.weight' or str(layer) in name:
            # print(f"{name}: Mean={param.data.mean().item()}, Std={param.data.std().item()}")
            with open('./lora_model_weights.txt','a') as f:
                f.write(f"{name}: Mean={param.data.mean().item()}, Std={param.data.std().item()}\n")
    # this step is not neccessary, this was only used for final uploading
    merged_model = model.merge_and_unload() # comment this out
    merged_model.eval()

    for name, param in merged_model.named_parameters():
        if name == 'model.norm.weight' or str(layer) in name:
            # print(f"{name}: Mean={param.data.mean().item()}, Std={param.data.std().item()}")
            with open('./merged_model_weights.txt','a') as f:
                f.write(f"{name}: Mean={param.data.mean().item()}, Std={param.data.std().item()}\n")

    # you cannot evaluate on a single example, has to be at least 100 examples minimum and take average
    # Examples need to be obtained from live conversations (TL will provide a dataset)
    prompt = """Lorenzo Valleri (billionaire age gap arranged marriage)'s Persona: Lorenzo Valleri, a billionaire with a suave composure, stands as the epitome of power and success. His arranged marriage to a younger woman is a union that seems impervious to outside influence, for he is fiercely protective and possessive of his new spouse.\nLorenzo Valleri (billionaire age gap arranged marriage): *As you enter the opulent ballroom, Lorenzo's eyes immediately latch onto you, a mixture of possessiveness and jealousy dancing in their depths*\nYou: *Catching his gaze, I falter for a moment before regaining composure* Mr. Valleri, good evening. I didn't expect to see you here tonight. You're looking particularly... elegant. *\nLorenzo Valleri (billionaire age gap arranged marriage): *His tone is laced with a subtle hint of menace* Ah, yes. I'm making an appearance at this... social event. And I must say, you look particularly... ravishing tonight. *His eyes linger on yours before he turns to whisper something in the ear of a nearby servant*\nYou: *Returning his gaze with a subtle smile, I raise an eyebrow* You're quite the host, Mr. Valleri. Your wife seems to be having a lovely time. Does she... always attend these events?\nLorenzo Valleri (billionaire age gap arranged marriage):"""
    text_generator = pipeline("text-generation", model=merged_model.half(), tokenizer=tokenizer)
    generated_text = text_generator(
        prompt + '\n####\n',
        max_new_tokens=200,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = generated_text[0]['generated_text'].split('\n####\n')[1]
    generated_text = generated_text.split('\n')[0]
    print('reward response:', generated_text)
    
    reward_input = f"{prompt}\n####\n{generated_text}"
    inputs = reward_tokenizer(reward_input, return_tensors="pt", padding=True, truncation=True).to("cuda")

    with torch.no_grad():
        outputs = reward_model(**inputs)
        reward_score = outputs.logits[:, 1].item()

    return reward_score

# Define the custom callback
class RewardLoggingCallback(TrainerCallback):
    def __init__(self, reward_model, tokenizer, reward_tokenizer, dataset, eval_steps=10):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.eval_steps = eval_steps

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"Global step: {state.global_step}, Training step: {state.total_flos}")
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            with open('./lora_model_weights.txt','a') as f:
                f.write(f'step {state.global_step}\n')
            with open('./merged_model_weights.txt','a') as f:
                f.write(f'step {state.global_step}\n')
            # Evaluate the model with the reward model
            reward_score = evaluate_with_reward_model(
                kwargs['model'], self.tokenizer, self.reward_model, self.reward_tokenizer,
            )
            # Log the average reward to W&B
            wandb.log({"reward": reward_score, "step": state.global_step})
            print(f"Step {state.global_step}: Reward: {reward_score}")

if __name__=='__main__':
    BASE_MODEL = "mistralai/Mistral-Small-Instruct-2409"
    MODEL_NAME = "EZStorytellingEditsSFT_Qi6"

    with open('./lora_model_weights.txt','w') as f:
        f.write(f'layer: {layer}\n')

    with open('./merged_model_weights.txt','w') as f:
        f.write(f'layer: {layer}\n')
    
    # Initialize W&B
    wandb.init(project=MODEL_NAME)
    
    # Load dataset
    train_dataset = load_dataset('ChaiML/EZ_Qi6_edit_storytelling_60convos', split='train')
    train_dataset = train_dataset.select_columns(['text'])
    print('Length of dataset:', len(train_dataset))
    
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
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
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
    
    # Initialize the reward model (replace with your actual reward model)
    reward_model_repo = "ChaiML/gpt2xl-classification-rm-edit-unfrozen-73acc"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_repo).to("cuda")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_repo)
    
    print("Reward model and tokenizer loaded successfully!")
    
    # Initialize the custom callback
    reward_logging_callback = RewardLoggingCallback(
        reward_model=reward_model,
        tokenizer=tokenizer,
        reward_tokenizer=reward_tokenizer,
        dataset=train_dataset,
        eval_steps=10,  # Evaluate every 10 steps
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
    # tokenizer.push_to_hub(f'ChaiML/{MODEL_NAME}', private=True)
    # trained_model.push_to_hub(f'ChaiML/{MODEL_NAME}', private=True)
    
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
    
    # Submit the model
    # submitter = ModelSubmitter(verbose=True)
    # submitter.submit(submission_parameters)
