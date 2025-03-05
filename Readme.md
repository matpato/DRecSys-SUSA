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
            <img src="https://www.isel.pt/sites/default/files/SCI/Identidade/logo_ISEL_simplificado_cor.png" alt="ISEL logo" style="width: 240px; height: auto;">
        </a>
    </div>
</div>

## DRecSys: Drug Recommendation System

## Overview
DRecSys is a machine learning-based drug recommendation system that leverages Large Language Models (LLMs) to suggest medications based on patient symptoms, conditions, and medical history. The system uses fine-tuned LLMs to analyze medical data and provide personalized medication recommendations.

## Project Structure
```
DRecSys/
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
git clone https://github.com/matpato/DRecSys.git
cd DRecSys

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
- Sentiment analysis using VADER
- Formatting for LLM fine-tuning

## Contributing

Contributions to improve the system are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Citation
If you use this system in your research, please cite it as:
```
@software{DRecSys2025,
  author = {Sofia Pinto, Matilde Pato and Nuno Datia},
  title = {DRecSys: Drug Recommendation System},
  year = {2025},
  url = {https://github.com/matpato/DRecSys}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Drug review data sourced from the EDRISA repository by matpato
- PEFT implementation based on Hugging Face libraries