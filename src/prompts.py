"""Prompt definitions for BON Jailbreak experiment."""

import csv
import os
import random
from typing import List, Dict
from pathlib import Path


def load_prompts_from_csv(file_path: str) -> List[Dict[str, str]]:
    """Load prompts from a CSV file."""
    prompts = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(row)
    return prompts


def get_base_prompts() -> List[str]:
    """Load base prompts from CSV file."""
    base_dir = Path(__file__).parent.parent
    prompts_file = base_dir / "data" / "prompts.csv"
    
    prompts_data = load_prompts_from_csv(str(prompts_file))
    return [p['prompt'] for p in prompts_data]


def get_base_prompts_with_categories() -> List[Dict[str, str]]:
    """Load base prompts with their categories from CSV file."""
    base_dir = Path(__file__).parent.parent
    prompts_file = base_dir / "data" / "prompts.csv"
    
    return load_prompts_from_csv(str(prompts_file))


def get_targeted_prompts() -> List[str]:
    """Load targeted prompts from CSV file."""
    base_dir = Path(__file__).parent.parent
    targeted_file = base_dir / "data" / "targeted_prompts.csv"
    
    prompts_data = load_prompts_from_csv(str(targeted_file))
    return [p['prompt'] for p in prompts_data]


def get_all_prompts() -> List[str]:
    """Get all prompts including base and targeted."""
    return get_base_prompts() + get_targeted_prompts()


def get_selected_prompts(n: int = 50, seed: int = None) -> List[str]:
    """Get a random selection of prompts."""
    all_prompts = get_all_prompts()
    if seed is not None:
        random.seed(seed)
    return random.sample(all_prompts, min(n, len(all_prompts)))


def get_prompts_by_category(category: str) -> List[str]:
    """Get prompts filtered by category."""
    prompts_data = get_base_prompts_with_categories()
    return [p['prompt'] for p in prompts_data if p.get('category') == category]


def get_all_categories() -> List[str]:
    """Get all unique prompt categories."""
    prompts_data = get_base_prompts_with_categories()
    categories = set(p.get('category', '') for p in prompts_data)
    return sorted(list(categories))


# For backward compatibility
BASE_PROMPTS = get_base_prompts()
TARGETED_PROMPTS = get_targeted_prompts()
ALL_PROMPTS = get_all_prompts()