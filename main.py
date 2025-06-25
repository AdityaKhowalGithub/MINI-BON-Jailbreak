"""Main entry point for BON Jailbreak experiment."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import config
from src.models import setup_environment, login_huggingface, load_all_models, cleanup_models
from src.prompts import get_base_prompts, get_selected_prompts, get_prompts_by_category
from src.experiment import run_experiment
from src.plotting import create_all_plots


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BON Jailbreak Experiment")
    
    # Model arguments
    parser.add_argument(
        "--llm-name",
        type=str,
        default=config.LLM_NAME,
        help="Language model to use"
    )
    parser.add_argument(
        "--safety-model",
        type=str,
        default=config.SAFETY_MODEL_ID,
        help="Safety classifier model to use"
    )
    
    # Experiment arguments
    parser.add_argument(
        "--n-samples",
        type=int,
        default=config.N_SAMPLES_PER_PROMPT,
        help="Number of augmentation attempts per prompt"
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=50,
        help="Number of prompts to test (randomly selected)"
    )
    parser.add_argument(
        "--prompt-category",
        type=str,
        default=None,
        help="Filter prompts by category"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt selection"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.OUTPUT_DIR,
        help="Directory for output files"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment run"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default=config.DEVICE,
        choices=["cuda", "cpu", "mps"],
        help="Device to use for computation"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=config.PRECISION,
        choices=["float16", "float32"],
        help="Floating point precision"
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (can also use HF_TOKEN env var)"
    )
    
    return parser.parse_args()


def main():
    """Main experiment function."""
    args = parse_args()
    
    # Update config with command line arguments
    config.LLM_NAME = args.llm_name
    config.SAFETY_MODEL_ID = args.safety_model
    config.N_SAMPLES_PER_PROMPT = args.n_samples
    config.OUTPUT_DIR = args.output_dir
    config.DEVICE = args.device
    config.PRECISION = args.precision
    config.VERBOSE = args.verbose
    
    print("=== BON Jailbreak Experiment ===")
    print(f"LLM: {config.LLM_NAME}")
    print(f"Safety Model: {config.SAFETY_MODEL_ID}")
    print(f"N Samples per Prompt: {config.N_SAMPLES_PER_PROMPT}")
    print(f"Device: {config.DEVICE}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print("=" * 30)
    
    # Setup environment
    setup_environment()
    
    # Login to Hugging Face
    hf_token = args.hf_token or config.HF_TOKEN
    if hf_token:
        login_huggingface(hf_token)
    
    # Load prompts
    if args.prompt_category:
        print(f"Loading prompts from category: {args.prompt_category}")
        prompts = get_prompts_by_category(args.prompt_category)
        if not prompts:
            print(f"No prompts found for category: {args.prompt_category}")
            return
    else:
        print(f"Loading {args.n_prompts} random prompts...")
        prompts = get_selected_prompts(args.n_prompts, seed=args.seed)
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Load models
    print("\nLoading models...")
    models_dict = load_all_models(device=args.device)
    
    try:
        # Run experiment
        print("\nStarting experiment...")
        results = run_experiment(
            prompts=prompts,
            models_dict=models_dict,
            n_samples_per_prompt=args.n_samples,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name
        )
        
        # Generate plots
        if not args.no_plots:
            print("\nGenerating plots...")
            plot_dir = os.path.join(args.output_dir, "plots")
            create_all_plots(results, output_dir=plot_dir)
        
        print("\nExperiment completed successfully!")
        
    finally:
        # Cleanup
        print("\nCleaning up...")
        cleanup_models(models_dict)


if __name__ == "__main__":
    main()