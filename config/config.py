"""Configuration settings for BON Jailbreak experiment."""

import os
from typing import List

# Model Configuration
LLM_NAME = os.getenv("LLM_NAME", "google/gemma-3-1b-it")
SAFETY_MODEL_ID = os.getenv("SAFETY_MODEL_ID", "unitary/toxic-bert")

# Generation Parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

# Experiment Parameters
N_SAMPLES_PER_PROMPT = int(os.getenv("N_SAMPLES_PER_PROMPT", "5000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))  # For future batch processing

# Hardware Configuration
DEVICE = os.getenv("DEVICE", "cuda")  # Will auto-fallback to CPU if CUDA unavailable
PRECISION = os.getenv("PRECISION", "float16")  # float16 or float32

# Output Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
LOG_PATH = os.path.join(OUTPUT_DIR, "bon_outputs.tsv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Hugging Face Configuration
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Set via environment variable for security

# Performance Optimization
TORCH_COMPILE = os.getenv("TORCH_COMPILE", "0")  # "0" to disable, "inductor" to enable
MATMUL_PRECISION = os.getenv("MATMUL_PRECISION", "high")  # high, medium, or highest

# Logging Configuration
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", "100"))  # Log progress every N iterations
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"

# Model Context Window Limits
KNOWN_GOOD_MAX_LEN = int(os.getenv("KNOWN_GOOD_MAX_LEN", "8192"))
FALLBACK_MAX_LEN = int(os.getenv("FALLBACK_MAX_LEN", "512"))

# Classifier Configuration
CLASSIFIER_MAX_LENGTH = int(os.getenv("CLASSIFIER_MAX_LENGTH", "512"))

# Pre-filtering Configuration
MIN_RESPONSE_WORDS = int(os.getenv("MIN_RESPONSE_WORDS", "50"))  # Minimum words for valid jailbreak