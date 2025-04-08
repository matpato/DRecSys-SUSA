<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div style="display: flex; align-items: center;">
    <div style="flex: 1;">
        <a href="https://isel.pt" target="_blank">
            <img src="./img/01_ISEL-Logotipo-RGB_Horizontal.png" alt="ISEL logo" style="width: 400px; height: auto;">
        </a>
    </div>
    <div style="flex: 3; text-align: left; padding-left: 20px;">
        <h3>DRecSys-SUSA: Drug Recommendation System Based on Symptoms and User Sentiment Analysis</h3>
    </div>
</div>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](https://www.docker.com/)
[![Llma](https://custom.typingmind.com/assets/models/llama.png)](https://www.llama.com)
[![Hugging Face](https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg)](https://huggingface.co)

## DRecSys: Drug Recommendation System

## Overview
DRecSys-SUSA is a machine learning-based drug recommendation system that leverages Large Language Models (LLMs) to suggest medications based on patient symptoms, conditions, and medical history. The system uses fine-tuned LLMs to analyze medical data and provide personalized medication recommendations.

## Project Structure
```
DRecSys-SUSA/
├── datasets.py          # Dataset construction and preprocessing
├── finetune.py          # LLM fine-tuning module
├── results.py           # Recommendation generation
├── evaluation.py        # Results analysis and visualization
├── run_drecsys.sh       # Complete pipeline execution script
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── data/                # Input data (CSV files)
├── configurations/      # Input pre-trained model and fine-tuned model
├── output/              # Generated outputs and models
└── plots/               # Generated plots
```

## Key Features
- **Two-Step Recommendation Pipeline**: Can extract relevant medical conditions before generating medication recommendations
- **PEFT Fine-tuning**: Uses Parameter-Efficient Fine-Tuning techniques (LoRA, QLoRA) to adapt LLMs efficiently
- **Multiple Ranking Methods**: Supports rankings based on original user ratings or VADER sentiment analysis
- **Comprehensive Evaluation**: Calculates standard information retrieval metrics (Hit Rate, MRR, NDCG)
- **Memory Optimization**: Uses model quantization to reduce memory requirements while maintaining performance
- **Automated Pipeline**: Includes a shell script (`run_drecsys.sh`) that orchestrates the entire workflow from data processing to evaluation

## Models
The system currently supports the following base models:
- **Llama 2 (7B parameters)**: `meta-llama/Llama-2-7b-chat-hf`
- **Llama 3 (8B parameters)**: `meta-llama/Meta-Llama-3-8B`

For semantic condition extraction, a smaller model is used:
- **Phi-3-mini**: `microsoft/Phi-3-mini-128k-instruct`

## Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.44+
- PEFT 0.12+
- bitsandbytes 0.41+
- Additional requirements specified in requirements.txt

## Installation
```bash
# Clone the repository
git clone git@github.com:matpato/DRecSys-SUSA.git
cd DRecSys-SUSA

# Using pip
pip install -r requirements.txt

# Make the shell script executable
chmod +x run.sh

# Using Docker
docker build -t drecsys -f Dockerfile .
```

## Usage

### Complete Pipeline with Shell Script
The included `run_drecsys.sh` shell script automates the entire pipeline:

```bash
# Make the script executable
chmod +x run_drecsys.sh

# Run the complete pipeline
./run_drecsys.sh
```

The script handles all steps in sequence:
1. Dataset processing
2. Model fine-tuning (can be skipped with `SKIP_FINETUNE=true ./run_drecsys.sh`)
3. Recommendation generation
4. Performance evaluation

Progress is displayed with clear indicators, and each step includes error checking.

### Individual Component Execution

#### Data Preparation
```python
# Run the dataset preparation script
python datasets.py
```

#### Fine-tuning the Model
```python
# Fine-tune the LLM on drug review data
python finetune.py
```

#### Generating Recommendations
```python
# Generate medication recommendations for test cases
python results.py
```

#### Evaluating Results
```python
# Analyze and visualize the recommendation performance
python evaluation.py
```

### Using Docker
```bash
# Run the complete pipeline in Docker
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output drecsys

# Skip fine-tuning when running in Docker
docker run -e SKIP_FINETUNE=true -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output drecsys
```

## Performance Metrics
The system is evaluated using standard recommendation system metrics:
- **Hit Rate (HR)**: Measures how often the top recommendation matches the user's preferred medication
- **Mean Reciprocal Rank (MRR)**: Evaluates the average position of relevant medications in the recommendation list
- **Normalized Discounted Cumulative Gain (NDCG)**: Assesses ranking quality with position-based discounting

## Dataset Sources
The system uses drug review datasets containing user ratings, medical conditions, and review text. Data processing includes:
- Removal of corrupted reviews
- Extraction of condition-to-drug mappings
- Sentiment analysis using <a href="https://github.com/cjhutto/vaderSentiment">VADER</a>
- Formatting for LLM fine-tuning

## Contributing

Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Limitations

- This is a proof of concept and should not replace professional medical advice
- Recommendations are limited by the quality and quantity of the training data
- The system does not account for individual patient factors such as age, gender, or medical history

## Citation
If you use this system in your research, please cite it as:
```
@software{DRecSys2025,
  author = {Sofia Pinto, Matilde Pato and Nuno Datia},
  title = {DRecSys: Drug Recommendation System},
  year = {2025},
  url = {https://github.com/matpato/DRecSys}
}

@mastersthesis{DRecSys-SUSA2025,
  author  = "Ana Sofia Pinto",
  title   = "Drug Recommendation System Based on Symptoms and User Sentiment Analysis",
  school  = "Instituto Superior de Engenharia de Lisboa",
  year    = "2025"
}
```

## Acknowledgments
- Drug review data sourced from the <a href="https://github.com/matpato/EDRISA.git">EDRISA</a> repository by matpato
- PEFT implementation based on Hugging Face libraries

Developed by Ana Sofia Pinto as part of the Drug Recommendation System Based on Symptoms and User Sentiment Analysis dissertation.

## License

[MIT License](LICENSE)