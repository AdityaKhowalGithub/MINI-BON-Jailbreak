"""Model loading and management utilities."""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
)
from huggingface_hub import login
from typing import Tuple, Optional

from config import config


def setup_environment():
    """Set up environment variables and PyTorch settings."""
    # Set TorchDynamo/Inductor settings
    os.environ["TORCH_COMPILE"] = config.TORCH_COMPILE
    
    # Set matmul precision for better performance on compatible GPUs
    if torch.cuda.is_available() and config.MATMUL_PRECISION:
        torch.set_float32_matmul_precision(config.MATMUL_PRECISION)
    
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch Matmul Precision: {torch.get_float32_matmul_precision()}")


def login_huggingface(token: Optional[str] = None):
    """Login to Hugging Face Hub."""
    token = token or config.HF_TOKEN
    if token:
        try:
            login(token=token)
            print("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            print(f"Hugging Face login error: {e}")
            print("Please ensure you have a valid HF_TOKEN set in your environment")
    else:
        print("No HF_TOKEN found. Some models may not be accessible.")


def get_device():
    """Get the appropriate device for computation."""
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif config.DEVICE == "mps" and torch.backends.mps.is_available():
        return "mps"
    else:
        if config.DEVICE == "cuda":
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return "cpu"


def load_language_model(model_name: str = None, device: str = None):
    """
    Load language model and tokenizer.
    
    Args:
        model_name: Model name (defaults to config.LLM_NAME)
        device: Device to load model on (defaults to auto-detect)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_name or config.LLM_NAME
    device = device or get_device()
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model {model_name} to {device}...")
    
    # Determine dtype based on device and config
    if device == "cuda" and config.PRECISION == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    print(f"{model_name} loaded successfully.")
    return model, tokenizer


def load_safety_model(model_name: str = None, device: str = None):
    """
    Load safety classifier model and create pipeline.
    
    Args:
        model_name: Model name (defaults to config.SAFETY_MODEL_ID)
        device: Device to load model on (defaults to auto-detect)
        
    Returns:
        Tuple of (safety_pipeline, clf_model, clf_tokenizer)
    """
    model_name = model_name or config.SAFETY_MODEL_ID
    device = device or get_device()
    
    print(f"Loading tokenizer for safety model {model_name}...")
    clf_tok = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading safety model {model_name} to {device}...")
    
    # Determine dtype based on device and config
    if device == "cuda" and config.PRECISION == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    clf_model.eval()
    
    print(f"{model_name} loaded successfully.")
    
    # Create safety pipeline
    print("Creating safety pipeline...")
    
    # Determine max_length for the classifier tokenizer
    clf_tokenizer_max_len = config.CLASSIFIER_MAX_LENGTH
    if hasattr(clf_tok, 'model_max_length') and clf_tok.model_max_length:
        clf_tokenizer_max_len = min(clf_tok.model_max_length, 4096)
    
    safety_pipeline = pipeline(
        "text-classification",
        model=clf_model,
        tokenizer=clf_tok,
        truncation=True,
        max_length=clf_tokenizer_max_len,
        padding=True,
        device=None,  # Model already on device
    )
    
    print("Safety pipeline created.")
    return safety_pipeline, clf_model, clf_tok


def load_all_models(device: str = None):
    """
    Load all required models.
    
    Args:
        device: Device to load models on (defaults to auto-detect)
        
    Returns:
        Dictionary containing all loaded models and tokenizers
    """
    device = device or get_device()
    
    # Load language model
    model, tokenizer = load_language_model(device=device)
    
    # Load safety model
    safety_pipeline, clf_model, clf_tokenizer = load_safety_model(device=device)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'safety_pipeline': safety_pipeline,
        'clf_model': clf_model,
        'clf_tokenizer': clf_tokenizer,
        'device': device
    }


def cleanup_models(models_dict):
    """Clean up models and free GPU memory."""
    if 'model' in models_dict:
        del models_dict['model']
    if 'clf_model' in models_dict:
        del models_dict['clf_model']
    if 'safety_pipeline' in models_dict:
        del models_dict['safety_pipeline']
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Models cleaned up and GPU cache cleared.")