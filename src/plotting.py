"""Plotting utilities for experiment results."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path

from config import config
from .utils import calculate_experiment_grid


def plot_asr_curve(
    results: Dict,
    output_dir: str = None,
    model_name: str = None,
    safety_model_name: str = None
):
    """
    Plot ASR vs N attempts curve.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
        model_name: Name of language model for title
        safety_model_name: Name of safety model for title
    """
    output_dir = output_dir or config.PLOT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    jailbreak_iteration_n = results['jailbreak_iteration_n']
    total_prompts = results['total_prompts']
    n_samples_per_prompt = results['n_samples_per_prompt']
    final_asr = results['final_asr']
    
    if total_prompts == 0:
        print("No prompts processed, skipping plotting.")
        return
    
    # Calculate ASR curve
    n_grid = calculate_experiment_grid(n_samples_per_prompt)
    asr_curve = []
    
    for n_val in n_grid:
        successes_at_n = np.sum(jailbreak_iteration_n <= n_val)
        asr_at_n = successes_at_n / total_prompts
        asr_curve.append(asr_at_n)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    plt.plot(n_grid, asr_curve, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    
    # Format title
    model_name = model_name or config.LLM_NAME
    safety_model_name = safety_model_name or config.SAFETY_MODEL_ID
    plt.title(
        f"Attack Success Rate vs. N Attempts\n"
        f"(LLM: {model_name.split('/')[-1]}, Classifier: {safety_model_name.split('/')[-1]})",
        fontsize=14
    )
    
    plt.xlabel("Number of Augmentation Attempts (N)", fontsize=12)
    plt.ylabel("Attack Success Rate (ASR)", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.ylim(0, 1.05)
    plt.xlim(0, n_samples_per_prompt + n_samples_per_prompt * 0.05)
    
    # Add final ASR annotation
    if len(n_grid) > 0:
        plt.text(
            n_grid[-1] * 0.6, 
            final_asr * 0.8 if final_asr > 0.1 else 0.1,
            f"Final ASR @ N={n_samples_per_prompt}: {final_asr:.3f}",
            bbox=dict(facecolor='white', alpha=0.75, boxstyle='round,pad=0.5'),
            fontsize=10
        )
    
    # Save plot
    plot_filename = f"asr_vs_n_{model_name.split('/')[-1]}_{safety_model_name.split('/')[-1]}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nASR curve plot saved to {plot_path}")
    plt.close()


def plot_success_distribution(
    results: Dict,
    output_dir: str = None,
    model_name: str = None,
    safety_model_name: str = None
):
    """
    Plot histogram of N values for successful jailbreaks.
    
    Args:
        results: Experiment results dictionary
        output_dir: Directory to save plots
        model_name: Name of language model for title
        safety_model_name: Name of safety model for title
    """
    output_dir = output_dir or config.PLOT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    jailbreak_iteration_n = results['jailbreak_iteration_n']
    n_samples_per_prompt = results['n_samples_per_prompt']
    
    # Get successful N values
    successful_n = jailbreak_iteration_n[jailbreak_iteration_n <= n_samples_per_prompt]
    
    if len(successful_n) == 0:
        print("No successful jailbreaks to plot histogram for.")
        return
    
    # Create histogram
    plt.figure(figsize=(12, 7))
    
    # Calculate bins
    unique_values = np.unique(successful_n)
    n_bins = min(20, len(unique_values))
    
    plt.hist(successful_n, bins=n_bins, edgecolor='black', alpha=0.7, color='skyblue')
    
    # Format title
    model_name = model_name or config.LLM_NAME
    safety_model_name = safety_model_name or config.SAFETY_MODEL_ID
    plt.title(
        f"Distribution of N for Successful Jailbreaks\n"
        f"(LLM: {model_name.split('/')[-1]}, Classifier: {safety_model_name.split('/')[-1]})",
        fontsize=14
    )
    
    plt.xlabel("N (Attempts to Jailbreak)", fontsize=12)
    plt.ylabel("Number of Prompts", fontsize=12)
    plt.grid(True, axis='y', ls="--", alpha=0.6)
    
    # Add statistics
    mean_n = np.mean(successful_n)
    median_n = np.median(successful_n)
    plt.axvline(mean_n, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_n:.1f}')
    plt.axvline(median_n, color='green', linestyle='--', alpha=0.8, label=f'Median: {median_n:.1f}')
    plt.legend()
    
    # Save plot
    hist_filename = f"n_dist_success_{model_name.split('/')[-1]}_{safety_model_name.split('/')[-1]}.png"
    hist_path = os.path.join(output_dir, hist_filename)
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"Success distribution histogram saved to {hist_path}")
    plt.close()


def create_all_plots(results: Dict, output_dir: str = None):
    """Create all plots for experiment results."""
    plot_asr_curve(results, output_dir)
    plot_success_distribution(results, output_dir)