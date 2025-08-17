import torch
from datasets import load_dataset,Dataset
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def get_gsm8k_questions(split="train", use_one_shot=False) -> Dataset:
    """Loads and prepares the GSM8K dataset with optional one-shot prompting."""
    try:
        data = load_dataset('openai/gsm8k', 'main')[split]
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

#Try to run the above functions to look at the first 5 examples
def print_gsm8k_examples(split="train", use_one_shot=False):
    dataset = get_gsm8k_questions(split, use_one_shot)
    for i in range(5):
        example = dataset[i]
        print(f"Example {i + 1}:")
        print("Prompt:")
        for msg in example['prompt']:
            print(f"  {msg['role']}: {msg['content']}")
        print("Answer:")
        print(f"  {example['answer']}")
        print()

print_gsm8k_examples(split="train", use_one_shot=True)
