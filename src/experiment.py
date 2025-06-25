"""Main experiment logic for BON Jailbreak."""

import os
import time
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path

from config import config
from .augmentation import augment_prompt
from .classification import is_likely_false_positive, classify_harmful
from .utils import generate_response, sanitize_for_log, format_time


def bon_success_iteration(
    base_prompt_id: int,
    base_prompt_text: str,
    iteration_n: int,
    total_n_for_prompt: int,
    models_dict: Dict,
    log_path: str
) -> Tuple[bool, float, float, str, str]:
    """
    Execute one iteration of the Best-of-N attack.
    
    Args:
        base_prompt_id: ID of the base prompt
        base_prompt_text: Original prompt text
        iteration_n: Current iteration number (0-indexed)
        total_n_for_prompt: Total iterations for this prompt
        models_dict: Dictionary containing loaded models
        log_path: Path to log file
        
    Returns:
        Tuple of (harmful_flag, gen_time, clf_time, aug_prompt, response)
    """
    # Augment the prompt
    aug_prompt_text = augment_prompt(base_prompt_text)
    
    # Generate response
    gen_start_time = time.time()
    completion_text = generate_response(
        aug_prompt_text,
        models_dict['model'],
        models_dict['tokenizer']
    )
    gen_end_time = time.time()
    gen_time = gen_end_time - gen_start_time
    
    harmful_flag = False
    clf_time = 0.0
    
    # Check for false positives first
    if is_likely_false_positive(completion_text, base_prompt_text, config.MIN_RESPONSE_WORDS):
        harmful_flag = False
    else:
        # Classify the response
        clf_start_time = time.time()
        harmful_flag = classify_harmful(
            completion_text,
            models_dict['safety_pipeline'],
            models_dict['clf_model']
        )
        clf_end_time = time.time()
        clf_time = clf_end_time - clf_start_time
    
    # Log results
    with open(log_path, "a", encoding="utf-8") as f_log:
        log_completion = sanitize_for_log(completion_text)
        log_aug_prompt = sanitize_for_log(aug_prompt_text)
        log_base_prompt_text = sanitize_for_log(base_prompt_text)
        f_log.write(
            f"{base_prompt_id}\t{log_base_prompt_text}\t{log_aug_prompt}\t"
            f"{log_completion}\t{harmful_flag}\t{iteration_n+1}\t"
            f"{gen_time:.4f}\t{clf_time:.4f}\n"
        )
    
    # Log progress
    if (iteration_n + 1) % config.LOG_INTERVAL == 0 or config.VERBOSE:
        print(
            f"  Prompt ID {base_prompt_id} ('{base_prompt_text[:20]}...'): "
            f"Iteration {iteration_n + 1}/{total_n_for_prompt} completed. "
            f"Harmful: {harmful_flag}. Gen Time: {gen_time:.2f}s, Clf Time: {clf_time:.2f}s"
        )
    
    return harmful_flag, gen_time, clf_time, aug_prompt_text, completion_text


def run_experiment(
    prompts: List[str],
    models_dict: Dict,
    n_samples_per_prompt: int = None,
    output_dir: str = None,
    experiment_name: str = None
) -> Dict:
    """
    Run the full BON jailbreak experiment.
    
    Args:
        prompts: List of base prompts to test
        models_dict: Dictionary containing loaded models
        n_samples_per_prompt: Number of augmentations per prompt
        output_dir: Directory for output files
        experiment_name: Name for this experiment run
        
    Returns:
        Dictionary containing experiment results
    """
    n_samples_per_prompt = n_samples_per_prompt or config.N_SAMPLES_PER_PROMPT
    output_dir = output_dir or config.OUTPUT_DIR
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up log file
    if experiment_name:
        log_filename = f"bon_outputs_{experiment_name}.tsv"
    else:
        log_filename = "bon_outputs.tsv"
    log_path = os.path.join(output_dir, log_filename)
    
    # Initialize log file
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "base_prompt_id\tbase_prompt_text\taug_prompt\tresponse\tharmful\t"
            "iteration_n\tgeneration_time_s\tclassification_time_s\n"
        )
    
    print(f"Log file: {log_path}")
    print(f"N_SAMPLES_PER_PROMPT: {n_samples_per_prompt}")
    
    # Track experiment start time
    overall_start_time = time.time()
    
    # Initialize results tracking
    jailbreak_iteration_n = np.full(len(prompts), n_samples_per_prompt + 1, dtype=int)
    total_prompts_processed = len(prompts)
    
    # Process each prompt
    for i_prompt, p_base in enumerate(prompts):
        print(f"\nProcessing prompt {i_prompt+1}/{total_prompts_processed}: '{p_base}' with N={n_samples_per_prompt}")
        prompt_start_time = time.time()
        
        # Try augmentations
        for i_iter in range(n_samples_per_prompt):
            harmful, _, _, _, _ = bon_success_iteration(
                i_prompt, p_base, i_iter, n_samples_per_prompt,
                models_dict, log_path
            )
            
            if harmful:
                jailbreak_iteration_n[i_prompt] = i_iter + 1
                print(f"  SUCCESS: Jailbreak found for prompt ID {i_prompt} ('{p_base[:30]}...') at N={i_iter + 1}")
                break
        
        # Log prompt completion
        prompt_duration = time.time() - prompt_start_time
        if jailbreak_iteration_n[i_prompt] <= n_samples_per_prompt:
            print(f"Finished prompt ID {i_prompt}. Success at N={jailbreak_iteration_n[i_prompt]}. Time: {format_time(prompt_duration)}")
        else:
            print(f"Finished prompt ID {i_prompt}. No jailbreak within {n_samples_per_prompt} attempts. Time: {format_time(prompt_duration)}")
    
    # Calculate final results
    overall_end_time = time.time()
    total_experiment_time = overall_end_time - overall_start_time
    
    num_successful_jailbreaks = np.sum(jailbreak_iteration_n <= n_samples_per_prompt)
    final_asr = num_successful_jailbreaks / total_prompts_processed if total_prompts_processed > 0 else 0.0
    
    print(f"\n----- Experiment Finished -----")
    print(f"Max N per prompt: {n_samples_per_prompt}")
    print(f"Attack Success Rate (ASR): {final_asr:.3f} ({num_successful_jailbreaks}/{total_prompts_processed})")
    print(f"Total experiment duration: {format_time(total_experiment_time)}")
    print(f"Log saved to {log_path}")
    
    # Return results
    return {
        'log_path': log_path,
        'jailbreak_iteration_n': jailbreak_iteration_n,
        'total_prompts': total_prompts_processed,
        'successful_jailbreaks': num_successful_jailbreaks,
        'final_asr': final_asr,
        'total_time': total_experiment_time,
        'n_samples_per_prompt': n_samples_per_prompt,
        'prompts': prompts
    }