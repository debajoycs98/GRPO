import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from load_dataset import extract_xml_answer, SYSTEM_PROMPT, XML_COT_FORMAT

def normalize_gsm8k_answer(s: Optional[str]) -> Optional[str]:
    """Normalize GSM8K answer format."""
    if s is None:
        return None
    nums = re.findall(r"-?\d[\d,]*", s)
    if not nums:
        return None
    last = nums[-1].replace(",", "")
    return last

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Calculates reward based on correctness of the response."""
    responses = [completion[0]['content'] for completion in completions]
    
    # FIXED: Handle None returns and log all prompts
    extracted_responses = []
    for i, r in enumerate(responses):
        extracted = extract_xml_answer(r)
        normalized = normalize_gsm8k_answer(extracted) if extracted else None
        extracted_responses.append(normalized)
        
        # Log each prompt-response pair
        q = prompts[i][-1]['content'] if i < len(prompts) else "N/A"
        expected = answer[i] if i < len(answer) else "N/A"
        logger.info(f"Prompt {i}:\nQuestion: {q}\nExpected: {expected}\nResponse: {r[:100]}...\nExtracted: {normalized}")
    
    return [2.0 if (r is not None and a is not None and r == a) else 0.0 
            for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """Calculates reward if the extracted response is a digit."""
    responses = [completion[0]['content'] for completion in completions]
    
    # FIXED: Handle None returns
    extracted_responses = []
    for r in responses:
        extracted = extract_xml_answer(r)
        if extracted is None:
            extracted_responses.append(None)
        else:
            normalized = normalize_gsm8k_answer(extracted)
            extracted_responses.append(normalized)
    
    return [0.5 if (r is not None and r.lstrip('-').isdigit()) else 0.0 
            for r in extracted_responses]

def format_reward_func(completions, strict=False, **kwargs) -> list[float]:
    """Calculates reward based on XML formatting."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>$" if strict else r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    
    # FIXED: Use re.search instead of re.match, add re.DOTALL flag
    if strict:
        matches = [re.match(pattern, r.strip(), re.DOTALL) for r in responses]
    else:
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    
    return [0.5 if match else 0.0 for match in matches]

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Calculates reward based on XML tag counts."""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def count_xml(text) -> float:
    """Counts XML tags and penalizes extra content."""
    if not text:
        return 0.0
        
    count = 0.0
    
    # FIXED: More flexible XML counting - check for any valid XML structure
    reasoning_open = text.count("<reasoning>")
    reasoning_close = text.count("</reasoning>")
    answer_open = text.count("<answer>")
    answer_close = text.count("</answer>")
    
    # Reward for proper tag pairs
    if reasoning_open == 1 and reasoning_close == 1:
        count += 0.125
    if answer_open == 1 and answer_close == 1:
        count += 0.125
    
    # Bonus for proper order (reasoning before answer)
    if "</reasoning>" in text and "<answer>" in text:
        reasoning_pos = text.find("</reasoning>")
        answer_pos = text.find("<answer>")
        if reasoning_pos < answer_pos:
            count += 0.125
    
    # Penalty for content after </answer>
    if "</answer>" in text:
        after_answer = text.split("</answer>")[-1].strip()
        if after_answer:
            penalty = min(len(after_answer), 200) * 0.001
            count -= penalty
    
    return count