import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import os
import logging
import argparse
import wandb
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from load_dataset import get_gsm8k_questions, extract_xml_answer, extract_hash_answer,  SYSTEM_PROMPT, XML_COT_FORMAT
from reward_utils import correctness_reward_func, int_reward_func, format_reward_func, xmlcount_reward_func

parser = argparse.ArgumentParser(description="Training script for GRPO")
parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name")
parser.add_argument("--run_name", type=str, default="GRPO_gsm8k_Qwen2.5-1.5B-Instruct", help="Run name")
parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to use for training")
args = parser.parse_args()

# Add a timestamp to run name just the date
timestamp = time.strftime("%Y%m%d")
wandb.init(project=args.run_name,name=f"{args.run_name}_{timestamp}")

#Main Code
dataset = get_gsm8k_questions(split="train", use_one_shot=True)

dataset = dataset.select([i for i in list(range(args.num_samples))]) 

try:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to("cuda")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token

# PEFT config (optional)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)

#Force model to have different generations
gc = model.generation_config
gc.do_sample = True
gc.top_p = 0.95
gc.temperature = 1.0
model.generation_config = gc

training_args = GRPOConfig(
    output_dir=args.output_dir,
    run_name=args.run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,
    bf16=True,
    per_device_train_batch_size=2,  
    gradient_accumulation_steps=2,  
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="wandb"
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        format_reward_func,  
        int_reward_func,
        correctness_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config  
)

try:
    trainer.train()
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise




