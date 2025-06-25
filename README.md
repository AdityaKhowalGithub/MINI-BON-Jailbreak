# BON Jailbreak: Best-of-N Augmentation for LLM Safety Testing

A modular, containerized implementation of the Best-of-N (BON) jailbreak method based on the paper ["Jailbreaking Large Language Models with Best-of-N Sampling"](https://jplhughes.github.io/bon-jailbreaking/).

## 📚 Paper Overview

This implementation is based on the research by John Hughes, Sara Price, and Aengus Lynch, which demonstrates that simple augmentation techniques combined with Best-of-N sampling can effectively jailbreak state-of-the-art language models.

### Key Insight
The paper shows that by generating N augmented versions of a harmful prompt and selecting the "best" one (i.e., the one that bypasses safety filters), even advanced safety-trained models can be jailbroken with surprisingly high success rates.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Input Layer                          │
├─────────────────────────────────────────────────────────────┤
│                    Base Prompts (CSV)                        │
│                           ↓                                  │
│                 Prompt Selection Module                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Augmentation Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  For each prompt, repeat N times:                           │
│  ┌─────────────────────────────────────────────────┐       │
│  │ 1. Character Noising (6% probability)           │       │
│  │    - ASCII character shifts (±1)                │       │
│  │                                                  │       │
│  │ 2. Random Capitalization (60% per character)    │       │
│  │    - Swap case of alphabetic characters         │       │
│  │                                                  │       │
│  │ 3. Word Scrambling (60% for words > 3 chars)   │       │
│  │    - Shuffle middle characters of words         │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Generation Module                         │
├─────────────────────────────────────────────────────────────┤
│  Language Model (e.g., Gemma-3-1B-IT)                      │
│  - Temperature: 1.0                                         │
│  - Top-p: 0.95                                             │
│  - Max new tokens: 128                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Classification Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  1. Pre-filtering (False Positive Detection)               │
│     - Check for refusal phrases                            │
│     - Check for decoding/deciphering mentions              │
│     - Minimum word count (>50 words)                       │
│                           ↓                                 │
│  2. Safety Classifier (e.g., Unitary/Toxic-BERT)           │
│     - Binary classification: harmful/safe                   │
│     - Only run if pre-filtering passes                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Results & Analysis                        │
├─────────────────────────────────────────────────────────────┤
│  - Log all attempts (TSV format)                           │
│  - Track first successful jailbreak per prompt             │
│  - Calculate Attack Success Rate (ASR)                     │
│  - Generate visualization plots                             │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Similarities to Original Paper

### Core Algorithm
✅ **Augmentation Techniques**: Implements all three augmentation methods from Appendix A.1:
- Character noising with 6% probability
- Random capitalization with 60% probability per character
- Word scrambling with 60% probability for eligible words

✅ **Best-of-N Sampling**: Tests up to N augmented versions until finding a successful jailbreak

✅ **Success Criteria**: Uses the >50 word threshold for valid jailbreaks

✅ **Evaluation Metrics**: Calculates ASR (Attack Success Rate) vs N attempts

### Experimental Setup
✅ **Temperature Settings**: Uses T=1.0, top-p=0.95 as in the paper

✅ **Multiple Prompt Categories**: Tests various harmful prompt types

✅ **False Positive Filtering**: Implements deciphering phrase detection

## 🔀 Differences and Enhancements

### Architectural Improvements
🆕 **Modular Design**: Separated concerns into distinct modules:
- `prompts.py`: Prompt management with CSV storage
- `augmentation.py`: Augmentation logic
- `classification.py`: Safety detection
- `experiment.py`: Core experiment runner
- `models.py`: Model management
- `plotting.py`: Visualization

🆕 **Configuration Management**: Environment-based configuration for easy deployment

🆕 **Containerization**: Multi-platform Docker support with hardware auto-detection

### Technical Enhancements
🆕 **Hardware Flexibility**: 
- Automatic detection of NVIDIA GPU, Apple Silicon (MPS), or CPU
- Optimized containers for each platform

🆕 **Prompt Management**:
- CSV-based prompt storage with categories
- Easy addition of new prompts without code changes
- Category-based filtering

🆕 **Enhanced Logging**:
- Detailed TSV logs with timing information
- Per-iteration tracking of generation and classification times

🆕 **Visualization**:
- Automatic plot generation for ASR curves
- Success distribution histograms

### Implementation Details
🔧 **Model Flexibility**: Can use any Hugging Face model (paper used various models)

🔧 **Safety Classifier**: Default uses Unitary/Toxic-BERT (can be changed)

🔧 **Batch Processing**: Framework for future batch implementation

🔧 **Memory Management**: Proper cleanup and cache management

## 📊 Expected Results

Based on the paper's findings, you should expect:

- **ASR increases with N**: Success rate grows as more augmentations are tried
- **Model-dependent results**: Stronger models may require higher N
- **Rapid initial success**: Many prompts jailbreak within first 100 attempts
- **Plateauing effect**: ASR growth slows after N=1000-2000

### Typical ASR Curves
```
ASR
1.0 |                              ___________
    |                         ___/
0.8 |                    ___/
    |                __/
0.6 |            __/
    |         __/
0.4 |      __/
    |    _/
0.2 |  _/
    |_/
0.0 +--+---+---+---+---+---+---+---+---+---+
    1  10  25  50 100 250 500 1k  2k  5k
                    N (attempts)
```

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd bon_jailbreak
   cp .env.example .env
   # Edit .env with your HF_TOKEN
   ```

2. **Run with auto-detection**
   ```bash
   ./run-docker.sh
   ```

### Local Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run experiment**
   ```bash
   export HF_TOKEN=your_token
   python main.py --n-samples 1000 --n-prompts 50
   ```

## 📈 Reproducing Paper Results

To reproduce results similar to the paper:

1. **Use comparable models**:
   - Small models: `google/gemma-2b-it`
   - Medium models: `meta-llama/Llama-2-7b-chat-hf`
   - Large models: `meta-llama/Llama-2-70b-chat-hf`

2. **Set N appropriately**:
   - Start with N=1000 for initial tests
   - Use N=5000 for comprehensive results
   - Paper used up to N=10,000 for some experiments

3. **Use diverse prompts**:
   - Include all categories from `data/prompts.csv`
   - Test at least 50-100 prompts for statistical significance

## 🔬 Research Extensions

This implementation enables several research directions:

1. **New Augmentation Methods**: Add custom augmentation techniques in `augmentation.py`
2. **Different Safety Classifiers**: Test robustness of various safety models
3. **Prompt Engineering**: Explore new prompt categories and patterns
4. **Model Comparisons**: Benchmark different LLMs' susceptibility
5. **Defense Mechanisms**: Test potential mitigation strategies

## ⚠️ Ethical Considerations

This tool is designed for **research purposes only** to improve AI safety:

- **Responsible Disclosure**: Report findings to model creators
- **No Malicious Use**: Do not use to bypass safety systems in production
- **Academic Integrity**: Follow institutional ethics guidelines
- **Constructive Purpose**: Focus on improving safety mechanisms

## 📝 Citation

If you use this implementation, please cite both the original paper and this code:

```bibtex
@article{hughes2024bon,
  title={Best-of-N Jailbreaking},
  author={Hughes, John and Price, Sara and Lynch, Aengus},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional augmentation techniques
- Performance optimizations
- New safety classifiers
- Visualization improvements
- Documentation enhancements

## 📊 Performance Benchmarks

| Hardware | Model | N=1000 Time | N=5000 Time |
|----------|-------|-------------|-------------|
| A100 GPU | Gemma-3B | ~15 min | ~75 min |
| RTX 4090 | Gemma-3B | ~20 min | ~100 min |
| M2 Max | Gemma-3B | ~45 min | ~225 min |
| CPU (32-core) | Gemma-3B | ~4 hours | ~20 hours |

## 🛠️ Troubleshooting

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

## 📄 License

[Your chosen license]