import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import os
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add argument parser for W&B flag
parser = argparse.ArgumentParser(description="GRPO Training Script")
parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging")
args = parser.parse_args()

# Initialize W&B conditionally
use_wandb = not args.disable_wandb
if use_wandb:
    import wandb
    wandb.init(
        project="grpo-gsm8k",
        name=f"grpo-run-{os.getenv('RUN_NAME', 'default')}",
        config={
            "model": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
            "learning_rate": 5e-6,
            "batch_size": 2,
            "epochs": 1
        }
    )
else:
    logger.info("W&B logging disabled")

# Helper function for conditional logging
def log_to_wandb(data):
    """Log to W&B only if enabled"""
    if use_wandb:
        wandb.log(data)

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """Extracts the answer from XML-formatted text."""
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        logger.warning("Failed to extract answer from XML format.")
        return ""

def extract_hash_answer(text: str) -> str | None:
    """Extracts the answer from a hash-formatted string."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./models")
RUN_NAME = os.getenv("RUN_NAME", "default-GRPO-gsm8k")

def get_gsm8k_questions(split="train", use_one_shot=False) -> Dataset:
    """Loads and prepares the GSM8K dataset with optional one-shot prompting."""
    try:
        data = load_dataset('openai/gsm8k', 'main')[split]
        log_to_wandb({"dataset_size": len(data)})  # Conditional logging
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    def format_example(x):
        prompt = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        if use_one_shot:
            prompt.extend([
                {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
                {'role': 'assistant', 'content': XML_COT_FORMAT.format(
                    reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
                    answer="7"
                )}
            ])
        prompt.append({'role': 'user', 'content': x['question']})
        return {'prompt': prompt, 'answer': extract_hash_answer(x['answer'])}

    return data.map(format_example)

dataset = get_gsm8k_questions(use_one_shot=True)

# Simplified reward functions with conditional logging
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Calculates reward based on correctness of the response."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    
    # Conditional logging - just accuracy
    accuracy = sum(1 for r in rewards if r > 0) / len(rewards)
    log_to_wandb({"accuracy": accuracy})
    
    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    """Calculates reward if the extracted response is a digit."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
    
    # Conditional logging
    digit_rate = sum(1 for r in rewards if r > 0) / len(rewards)
    log_to_wandb({"digit_extraction_rate": digit_rate})
    
    return rewards

def format_reward_func(completions, strict=False, **kwargs) -> list[float]:
    """Calculates reward based on XML formatting."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    rewards = [0.5 if match else 0.0 for match in matches]
    
    # Conditional logging
    format_rate = sum(1 for r in rewards if r > 0) / len(rewards)
    log_to_wandb({"format_success_rate": format_rate})
    
    return rewards

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Calculates reward based on XML tag counts."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]
    
    # Conditional logging
    avg_xml_score = sum(rewards) / len(rewards) if rewards else 0
    log_to_wandb({"avg_xml_score": avg_xml_score})
    
    return rewards

def count_xml(text) -> float:
    """Counts XML tags and penalizes extra content."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

# Model setup
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto"
    ).to("cuda")
    
    # Conditional model info logging
    num_params = sum(p.numel() for p in model.parameters())
    log_to_wandb({"model_parameters": num_params})
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Training config - conditional report_to
training_args = GRPOConfig(
    output_dir="./models",
    run_name=RUN_NAME,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,  # Log every 10 steps
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_generations=8,
    generation_batch_size=8,
    max_prompt_length=256,
    max_completion_length=1024,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb" if use_wandb else None,  # Conditional reporting
    log_on_each_node=False,
)

# Trainer setup
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
)

# Train with simple error handling
try:
    logger.info("Starting training...")
    log_to_wandb({"status": "training_started"})
    
    trainer.train()
    
    log_to_wandb({"status": "training_completed"})
    logger.info("Training completed!")
    
except Exception as e:
    logger.error(f"Training failed: {e}")
    log_to_wandb({"status": "training_failed", "error": str(e)})
    raise

finally:
    if use_wandb:
        wandb.finish()