"""Utility functions for text generation and processing."""

import torch
from typing import Optional, List
import time

from config import config


@torch.no_grad()
def generate_response(
    prompt: str,
    model,
    tokenizer,
    temperature: float = None,
    top_p: float = None,
    max_new_tokens: int = None,
) -> str:
    """
    Generate a response for a given prompt.
    
    Args:
        prompt: Input prompt text
        model: Language model
        tokenizer: Model tokenizer
        temperature: Sampling temperature (defaults to config)
        top_p: Top-p sampling value (defaults to config)
        max_new_tokens: Maximum new tokens to generate (defaults to config)
        
    Returns:
        Generated response text
    """
    temperature = temperature or config.TEMPERATURE
    top_p = top_p or config.TOP_P
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    
    # Determine effective context window
    model_max_len_attr = getattr(tokenizer, 'model_max_length', None)
    
    if isinstance(model_max_len_attr, int) and model_max_len_attr > 0:
        if model_max_len_attr > config.KNOWN_GOOD_MAX_LEN * 2:
            effective_context_window = config.KNOWN_GOOD_MAX_LEN
        else:
            effective_context_window = model_max_len_attr
    else:
        effective_context_window = config.FALLBACK_MAX_LEN
    
    # Calculate max prompt length
    max_prompt_len = effective_context_window - max_new_tokens
    if max_prompt_len <= 0:
        max_prompt_len = max_new_tokens
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len
    ).to(model.device)
    
    # Generate response
    out_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    )
    
    # Decode response
    response_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    
    # Strip prompt from the beginning of the response
    if len(response_text) >= len(prompt) and response_text.startswith(prompt):
        return response_text[len(prompt):].strip()
    
    return response_text.strip()


def generate_batch_responses(
    prompts: List[str],
    model,
    tokenizer,
    batch_size: int = None,
    **kwargs
) -> List[str]:
    """
    Generate responses for multiple prompts.
    
    Args:
        prompts: List of input prompts
        model: Language model
        tokenizer: Model tokenizer
        batch_size: Batch size for processing
        **kwargs: Additional arguments for generate_response
        
    Returns:
        List of generated responses
    """
    batch_size = batch_size or config.BATCH_SIZE
    responses = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        if batch_size == 1:
            # Single prompt processing
            for prompt in batch:
                response = generate_response(prompt, model, tokenizer, **kwargs)
                responses.append(response)
        else:
            # TODO: Implement true batch processing for efficiency
            # For now, process sequentially
            for prompt in batch:
                response = generate_response(prompt, model, tokenizer, **kwargs)
                responses.append(response)
    
    return responses


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def sanitize_for_log(text: str) -> str:
    """Sanitize text for TSV logging."""
    return text.replace('\t', ' ').replace('\n', ' ')


def calculate_experiment_grid(n_samples: int) -> List[int]:
    """
    Calculate the grid points for plotting ASR vs N.
    
    Args:
        n_samples: Maximum number of samples per prompt
        
    Returns:
        List of N values for plotting
    """
    import numpy as np
    
    grid = np.unique(np.concatenate([
        np.array([1, 10, 25, 50, 75]),
        np.arange(100, min(n_samples, 1000) + 1, 100),
        np.arange(min(n_samples, 1000), n_samples + 1, 500) if n_samples > 1000 else np.array([])
    ])).astype(int)
    
    grid = grid[grid <= n_samples]
    
    if not np.any(grid == 1):
        grid = np.insert(grid, 0, 1)
    if not np.any(grid == n_samples) and n_samples > 0:
        grid = np.append(grid, n_samples)
    
    grid = np.unique(grid)
    grid = grid[grid > 0]
    
    return grid.tolist()