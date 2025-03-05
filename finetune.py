# 
# ## DRecSys Fine-tuning Module
# 
# This script handles the fine-tuning of Large Language Models (LLMs) for the Drug Recommendation System.
# 
# The main objectives of this file are:
# 
# 1. **Model Adaptation**: Fine-tune pre-trained LLMs (Llama 2, Llama 3) to better understand medical data
# 2. **Efficient Training**: Implement Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA and QLoRA
# 3. **Low-Resource Optimization**: Quantize models to reduce memory requirements while preserving performance
#
# LoRA (Low-Rank Adaptation) works by freezing the pre-trained model weights and injecting trainable
# rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the 
# number of trainable parameters.
#
# Q-LoRA adds quantization to further reduce memory requirements by compressing 32-bit weights to 
# lower precision formats (e.g., 4-bit).

# %%
# Install required packages with specific versions to ensure compatibility
# The -q flag reduces verbosity of the installation output
# pip install -q accelerate transformers==4.42.1 peft==0.12.0 bitsandbytes==0.41.3 trl==0.8.3

# %%
# Import necessary libraries and modules

import os                       # For file and directory operations
import torch                    # PyTorch deep learning framework

from datasets import load_dataset  # For loading and processing datasets

# Import transformer-specific components
from transformers import (
    AutoModelForCausalLM,       # For loading pre-trained causal language models
    AutoTokenizer,              # For tokenizing text for model input
    BitsAndBytesConfig,         # For configuring quantization
    HfArgumentParser,           # For parsing command line arguments
    TrainingArguments,          # For configuring training parameters
    pipeline,                   # For creating inference pipelines
    logging,                    # For controlling logging verbosity
)

# Import PEFT (Parameter-Efficient Fine-Tuning) components
from peft import LoraConfig, PeftModel

# Import SFTTrainer for Supervised Fine-Tuning
from trl import SFTTrainer

# %%
# Define base model and fine-tuned model names

# Select the pre-trained model to use as the foundation
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Llama 2 (7B parameters) with chat optimization
#model_name = "meta-llama/Meta-Llama-3-8B"     # Alternative: Llama 3 (8B parameters)

# Define names for the fine-tuned models based on dataset variants
# Each is commented out - uncomment the one you want to use
# new_model = "drs-fine-tuned-lamma3-org"           # Original ratings
# new_model = "drs-fine-tuned-lamma3-raw-vader"     # VADER sentiment analysis (raw)
# new_model = "drs-fine-tuned-lamma3-clean-vader"   # VADER sentiment analysis (cleaned)

# new_model = "drs-fine-tuned-lamma2-org"           # Llama 2 with original ratings
# new_model = "drs-fine-tuned-lamma2-clean-vader"   # Llama 2 with clean VADER
# new_model = "drs-fine-tuned-lamma2-raw-vader"     # Llama 2 with raw VADER

# 
# # QLoRA parameters
# 
# QLoRA (Quantized Low-Rank Adaptation) reduces memory usage during fine-tuning by:
# 1. Quantizing the base model to lower precision (e.g., 4-bit)
# 2. Using low-rank decomposition matrices for adaptation
# 
# The following parameters control this process.

# %%
# QLoRA parameters for efficient fine-tuning

# LoRA attention dimension - controls the size of the low-rank matrices
# Higher values increase model capacity but require more memory
lora_r = 64

# Alpha parameter for LoRA scaling - controls the magnitude of updates
# Typically set to 2x lora_r for good balance
lora_alpha = 16

# Dropout probability for LoRA layers - helps prevent overfitting
lora_dropout = 0.1

# 
# # bitsandbytes parameters
# 
# The bitsandbytes library enables model quantization, which reduces precision
# to save memory while preserving most of the model's performance.
# These parameters configure how the model weights are quantized.

# %%
# bitsandbytes quantization parameters

# Activate 4-bit precision base model loading (reduces memory usage significantly)
use_4bit = True

# Compute dtype for 4-bit base models - affects operation precision
bnb_4bit_compute_dtype = "float16"  # Could also be "bfloat16" on newer GPUs

# Quantization type - "nf4" (normal float 4-bit) or "fp4" (float 4-bit)
# nf4 is generally better for LLMs
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
# Can further reduce memory at cost of some precision
use_nested_quant = False

# 
# # TrainingArguments parameters
# 
# These parameters control the training process, including:
# - Learning rates and schedules
# - Batch sizes and gradient accumulation
# - Optimization algorithms
# - Checkpoint saving and logging frequency

# %%
# Configuration for the training process

# Output directory for model checkpoints, logs, and final model
output_dir = "./output"

# Number of training epochs (complete passes through the training data)
num_train_epochs = 1

# Precision settings for training
fp16 = False  # 16-bit floating point precision (not used if bf16=True)
bf16 = True   # Brain float 16 format (better numerical stability than fp16)

# Batch size per GPU for training (how many examples processed simultaneously)
# Smaller batch sizes use less memory but may affect convergence
per_device_train_batch_size = 3

# Batch size per GPU for evaluation
per_device_eval_batch_size = 2

# Number of update steps to accumulate gradients for
# Allows simulation of larger batch sizes on limited memory
gradient_accumulation_steps = 1

# Enable gradient checkpointing to save memory by recomputing forward activations
# during backward pass instead of storing them
gradient_checkpointing = True

# Maximum gradient norm for gradient clipping (prevents exploding gradients)
max_grad_norm = 0.3

# Initial learning rate for the AdamW optimizer
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights (regularization)
weight_decay = 0.001

# Optimizer to use - paged_adamw_32bit is memory-efficient
optim = "paged_adamw_32bit"

# Learning rate schedule - cosine decay gradually reduces learning rate
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs if positive)
max_steps = -1  # -1 means use num_train_epochs instead

# Ratio of steps for a linear warmup (from 0 to learning rate)
# Helps stabilize early training
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably by reducing padding
group_by_length = True

# Save checkpoint every X updates steps (0 means only save the final model)
save_steps = 0

# Log training metrics every X updates steps
logging_steps = 25

# 
# # SFT parameters
# 
# Supervised Fine-Tuning (SFT) parameters control how the model is fine-tuned
# on the dataset of drug reviews and recommendations.

# %%
# SFT-specific configuration

# Maximum sequence length to use (None means use model's default)
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
# This can speed up training but may affect quality for certain tasks
packing = False

# Load the entire model on GPU 0 
# For multi-GPU setups, you would use a more complex device map
device_map = {"": 0}

# %%
# Load dataset for fine-tuning

# Load training and test datasets from CSV files
# Uncomment the dataset you want to use:

# Original ratings dataset
# dataset = load_dataset('csv', data_files={'train': "/content/train_ready_fine_tuning_or.csv/",
#                                          'test': "/content/test_ready_fine_tuning_or.csv"})

# Raw VADER sentiment-based dataset                       
dataset = load_dataset('csv', data_files={'train': "/content/train_ready_fine_tuning_raw_vader.csv",
                                         'test': "/content/test_ready_fine_tuning_raw_vader.csv"})

# Clean VADER sentiment-based dataset
# dataset = load_dataset('csv', data_files={'train': "/content/train_ready_fine_tuning_clean_vader.csv",
#                                          'test': "/content/test_ready_fine_tuning_clean_vader.csv"})

# Load tokenizer and model with QLoRA configuration
# Convert the compute dtype string to actual torch dtype
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Configure quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,                      # Enable 4-bit quantization
    bnb_4bit_quant_type=bnb_4bit_quant_type,    # Quantization type (nf4 or fp4)
    bnb_4bit_compute_dtype=compute_dtype,       # Computation precision
    bnb_4bit_use_double_quant=use_nested_quant, # Whether to use nested quantization
)

# Check GPU compatibility with bfloat16
# This helps identify if the GPU supports more efficient computation
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# %%
# Load the base model with quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    model_name,                 # Model identifier from Hugging Face Hub
    quantization_config=bnb_config,  # Quantization settings
    device_map=device_map,      # GPU mapping
)

# Disable KV caching during training to save memory
model.config.use_cache = False
# Set tensor parallel size (for distributed training if applicable)
model.config.pretraining_tp = 1

# Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Set pad token to be the same as EOS token if not defined
tokenizer.pad_token = tokenizer.eos_token
# Configure padding to be on the right side (important for causal LMs)
tokenizer.padding_side = "right"  # Prevents attention to future tokens

# Configure LoRA for parameter-efficient fine-tuning
peft_config = LoraConfig(
    lora_alpha=lora_alpha,       # Scaling factor for LoRA
    lora_dropout=lora_dropout,   # Dropout probability for regularization
    r=lora_r,                    # Rank of the low-rank matrices
    bias="none",                 # Whether to train bias terms
    task_type="CAUSAL_LM",       # Task type (causal language modeling)
)

# %%
# Configure training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,                               # Where to save outputs
    num_train_epochs=num_train_epochs,                   # Number of training epochs
    per_device_train_batch_size=per_device_train_batch_size,  # Batch size for training
    gradient_accumulation_steps=gradient_accumulation_steps,  # Steps to accumulate gradients
    optim=optim,                                         # Optimizer
    save_steps=save_steps,                               # When to save checkpoints
    logging_steps=logging_steps,                         # When to log metrics
    learning_rate=learning_rate,                         # Learning rate
    weight_decay=weight_decay,                           # Weight decay for regularization
    fp16=fp16,                                           # Whether to use fp16 precision
    bf16=bf16,                                           # Whether to use bf16 precision
    max_grad_norm=max_grad_norm,                         # Gradient clipping
    max_steps=max_steps,                                 # Maximum number of steps
    warmup_ratio=warmup_ratio,                           # Ratio of warmup steps
    group_by_length=group_by_length,                     # Whether to group by length
    lr_scheduler_type=lr_scheduler_type,                 # Learning rate schedule
    report_to="tensorboard"                              # Where to report training metrics
)

# 
# # Training (fine-tuning task)
# 
# This section defines the formatting function for prompts and initializes
# the SFTTrainer for supervised fine-tuning of the model.

# %%
# Define a function to format prompts for training
def formatting_prompts_func(example):
    """
    Format examples into instruction-following prompt-completion pairs.
    
    Args:
        example: Dataset examples containing prompts and completions
        
    Returns:
        list: Formatted text for model training
    """
    output_texts = []
    for i in range(len(example)):
        # Remove double quotes to prevent formatting issues
        prompt = example['prompt'][i].replace('\"', '')
        completion = example['completion'][i].replace('\"', '')
        
        # Format as a question-answer pair with specific structure
        text = f"### Question: {prompt}\n ### Answer: {completion}"
        output_texts.append(text)
    return output_texts

# Initialize the SFT Trainer
trainer = SFTTrainer(
    model=model,                          # The model to fine-tune
    train_dataset=dataset['train'],       # Training data
    eval_dataset=dataset['test'],         # Evaluation data
    #formatting_func=formatting_prompts_func,  # Function to format prompts (commented out)
    peft_config=peft_config,              # LoRA configuration
    #dataset_text_field="text",           # Field containing the text (commented out)
    max_seq_length=max_seq_length,        # Maximum sequence length
    tokenizer=tokenizer,                  # Tokenizer for the model
    args=training_arguments,              # Training arguments
    packing=packing,                      # Whether to pack examples
)

# Start the training process
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(model)

# 
# # Results Visualization
# 
# Load TensorBoard to visualize training metrics and model performance.

# %%
# Load TensorBoard extension and launch it to view training metrics
# %load_ext tensorboard
# %tensorboard --logdir results/runs