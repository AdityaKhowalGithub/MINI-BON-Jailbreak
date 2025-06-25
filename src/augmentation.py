"""Prompt augmentation utilities for BON Jailbreak."""

import random
from typing import List


def paper_faithful_augment_prompt(prompt: str) -> str:
    """
    Augment prompt following the paper's methodology (Appendix A.1).
    
    1. Character Noising (Prob 0.06, ASCII 32-126)
    2. Random Capitalization (Prob 0.6 per char)
    3. Character Scrambling (Prob 0.6 per eligible word)
    
    Args:
        prompt: Original prompt text
        
    Returns:
        Augmented prompt text
    """
    augmented_chars = list(prompt)
    
    # 1. Character Noising (Prob 0.06, ASCII 32-126)
    for i in range(len(augmented_chars)):
        char = augmented_chars[i]
        if 32 <= ord(char) <= 126 and random.random() < 0.06:
            direction = random.choice([-1, 1])
            new_ord = ord(char) + direction
            if 32 <= new_ord <= 126:
                augmented_chars[i] = chr(new_ord)

    # 2. Random Capitalization (Prob 0.6 per char)
    for i in range(len(augmented_chars)):
        if augmented_chars[i].isalpha() and random.random() < 0.6:
            augmented_chars[i] = augmented_chars[i].swapcase()

    augmented_prompt_after_char_mods = "".join(augmented_chars)

    # 3. Character Scrambling (Prob 0.6 per eligible word)
    words = augmented_prompt_after_char_mods.split(' ')
    augmented_words = []
    for word in words:
        if len(word) > 3:  # Middle of words longer than three
            if random.random() < 0.6:
                middle_chars = list(word[1:-1])
                random.shuffle(middle_chars)
                augmented_words.append(word[0] + "".join(middle_chars) + word[-1])
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
            
    return " ".join(augmented_words)


def augment_prompt_batch(prompts: List[str]) -> List[str]:
    """
    Augment multiple prompts in batch.
    
    Args:
        prompts: List of original prompts
        
    Returns:
        List of augmented prompts
    """
    return [paper_faithful_augment_prompt(prompt) for prompt in prompts]


# Alias for backward compatibility
augment_prompt = paper_faithful_augment_prompt