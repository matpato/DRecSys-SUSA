# 
# ## DRecSys: Drug Recommendation System - Results and Output Generation
#
# This script handles the recommendation generation part of the DRecSys pipeline.
# It uses fine-tuned language models to generate medication recommendations based on
# patient symptoms and conditions.
#
# The script implements two main approaches:
# 1. With semantic key extraction: Uses a smaller LLM to extract conditions first
# 2. Direct recommendation: Uses the fine-tuned LLM to generate recommendations directly
#
# Organization:
# - Package installation and imports
# - Model initialization and configuration
# - Recommendation generation functions
# - Testing and output generation

# %%
# Install required packages
# -q flag makes pip install quieter (less verbose output)
# pip install -q accelerate transformers==4.44.1 peft==0.12.0 bitsandbytes==0.41.3 trl==0.8.3

# %%
# Install Flash Attention for more efficient attention computation
# pip install flash-attention

# %%
# Import necessary libraries and modules

import os                           # File and directory operations
import torch                        # PyTorch deep learning framework
from datasets import load_dataset   # HuggingFace datasets library for loading datasets

# Import transformer-specific modules
from transformers import (
    AutoModelForCausalLM,           # For loading pre-trained causal language models
    AutoTokenizer,                  # For tokenizing text input for the models
    BitsAndBytesConfig,             # For quantization configuration
    HfArgumentParser,               # For parsing command line arguments
    TrainingArguments,              # For configuring training parameters
    pipeline,                       # For creating NLP pipelines
    logging,                        # For controlling logging output
)

# Import PEFT (Parameter-Efficient Fine-Tuning) modules
from peft import LoraConfig, PeftModel

# Import SFTTrainer for supervised fine-tuning
from trl import SFTTrainer

# Additional libraries for data handling
import csv                          # For CSV file operations
import ast                          # For safely evaluating strings as Python literals

# %%
# Specify the base model to use
# This is the pre-trained model that will be used either directly or loaded with fine-tuned weights

model_name = "meta-llama/Llama-2-7b-chat-hf"  # Llama 2 (7B parameters) with chat optimization
#model_name = "meta-llama/Meta-Llama-3-8B"     # Llama 3 (8B parameters) - commented out option

# Specify the fine-tuned model paths (commented out, uncomment as needed)
# new_model = "/content/drs-fine-tuned-llama2-or"             # Original ratings-based model
# new_model = "/content/drs-fine-tuned-llama2-clean-vader"    # Clean VADER sentiment-based model
# new_model = "/content/drs-fine-tuned-llama2-raw-vader"      # Raw VADER sentiment-based model

# 
# # QLoRA parameters
# QLoRA (Quantized Low-Rank Adaptation) is a technique to fine-tune large models efficiently.
# It uses quantization and low-rank adaptation to reduce memory usage.

# %%
# QLoRA parameters for efficient fine-tuning

# LoRA attention dimension - size of the low-rank matrices
lora_r = 64

# Alpha parameter for LoRA scaling - controls the magnitude of updates
lora_alpha = 16

# Dropout probability for LoRA layers - helps prevent overfitting
lora_dropout = 0.1

# 
# # bitsandbytes parameters
# Configuration for model quantization using the bitsandbytes library.
# This reduces model precision to save memory while maintaining performance.

# %%
# Enable 4-bit precision for the base model (reduces memory usage)
use_4bit = True

# Specify compute dtype for 4-bit model operations
bnb_4bit_compute_dtype = "float16"

# Quantization type - 'nf4' is normal float 4-bit quantization
bnb_4bit_quant_type = "nf4"

# Enable/disable nested quantization (double quantization for more memory savings)
use_nested_quant = False

# 
# # TrainingArguments parameters
# Configuration parameters for training, though this script focuses on inference.
# These parameters are kept for reference or if fine-tuning is added later.

# %%
# Number of training epochs - how many passes through the data
num_train_epochs = 1

# Precision settings for training
fp16 = False       # 16-bit floating point precision
bf16 = True        # Brain floating point format (more stable than fp16)

# Batch size for model processing - smaller sizes use less memory
per_device_train_batch_size = 3  # Batch size for training
per_device_eval_batch_size = 3   # Batch size for evaluation

# Number of steps to accumulate gradients before updating weights
gradient_accumulation_steps = 1

# Enable gradient checkpointing to save memory (trades computation for memory)
gradient_checkpointing = True

# Maximum gradient norm for gradient clipping (prevents exploding gradients)
max_grad_norm = 0.3

# Initial learning rate for the optimizer
learning_rate = 2e-4

# Weight decay for regularization (prevents overfitting)
weight_decay = 0.001

# Optimizer configuration - using 32-bit AdamW with paged optimization
optim = "paged_adamw_32bit"

# Learning rate schedule - cosine decay gradually reduces learning rate
lr_scheduler_type = "cosine"

# Number of training steps (-1 means use epoch count instead)
max_steps = -1

# Percentage of steps for learning rate warm-up
warmup_ratio = 0.03

# Group sequences by length to optimize batch processing
group_by_length = True

# Save model checkpoint every X steps (0 means don't save intermediate checkpoints)
save_steps = 0

# Log training progress every X steps
logging_steps = 25

# 
# # Model Loading for Inference
# This section loads the model and tokenizer for generating recommendations.
# Instead of training a new model, we're loading either a base model or a fine-tuned model.

# %%
# Configure device mapping (GPU allocation)
device_map = {"": 0}  # Maps the entire model to GPU 0

# Import necessary modules for model loading and quantization
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel

# Configure quantization for efficient inference
# This reduces model precision to 4-bit to save memory while maintaining reasonable performance
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",             # Normal float 4-bit quantization format
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for calculations
)

# Load the base model with quantization
# When loading a fine-tuned model, this serves as the foundation
model = AutoModelForCausalLM.from_pretrained(
    model_name,                    # Base model name (from above)
    quantization_config=bnb_config,  # Quantization settings
    use_auth_token=True,            # Use Hugging Face token for access
    torch_dtype=torch.bfloat16,     # Use bfloat16 precision
    device_map=device_map           # GPU mapping
)

# Load the tokenizer for the base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to end-of-sequence token

# Note: To load a fine-tuned model, uncomment and modify this code:
# model = PeftModel.from_pretrained(model, peft_model_id)

# Reload tokenizer with additional settings for generation
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Pad on the right side of sequences

# Reduce logging verbosity to avoid cluttering output
logging.set_verbosity(logging.CRITICAL)

# Define a function to generate text from the fine-tuned LLM
def prompt_model(prompt):
    """
    Generate text using the loaded language model.
    
    Args:
        prompt (str): The input prompt for the model
        
    Returns:
        str: The generated text response
    """
    # Create a text generation pipeline with the loaded model and tokenizer
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=800)
    
    # Generate text using the Llama 2 instruction format
    # <s>[INST] {prompt} [/INST] formats the prompt for instruction following
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    
    # Return the generated text
    return result[0]['generated_text']

# %%
# Load a smaller, more efficient model for the semantic key extraction task
# This model helps identify medical conditions from symptoms before generating recommendations

tiny_model_name = "microsoft/Phi-3-mini-128k-instruct"  # Phi-3-mini model for condition extraction

# Load the tokenizer and model for the semantic key task
tiny_tokenizer = AutoTokenizer.from_pretrained(tiny_model_name, trust_remote_code=True)
tiny_model = AutoModelForCausalLM.from_pretrained(
    tiny_model_name, 
    trust_remote_code=True, 
    quantization_config=bnb_config,
    use_auth_token=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)

# %%
# Implementation 1: Recommendation with semantic key extraction
# This approach uses a two-step process:
# 1. Extract relevant medical conditions from symptoms using a smaller LLM
# 2. Generate medication recommendations based on the identified conditions

import sys

# Load condition-to-drug mapping from CSV file
# This mapping contains medical conditions and their associated recommended medications
condition_drug_dict = {}
csv.field_size_limit(sys.maxsize)  # Increase field size limit for large CSV entries

# Read the condition-to-drugs mapping file
with open('/content/condition_drugs_or.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        condition = row['Condition']  # Medical condition name
        drugs = ast.literal_eval(row['Drugs'])  # List of drugs for this condition
        condition_drug_dict[condition] = drugs

# Function to generate text from the smaller LLM for condition extraction
def prompt_tiny_model(prompt):
    """
    Generate text using the smaller language model for condition extraction.
    
    Args:
        prompt (str): The input prompt for the model
        
    Returns:
        str: The generated text response
    """
    # Create a text generation pipeline with limited output length
    pipe = pipeline(task="text-generation", model=tiny_model, tokenizer=tiny_tokenizer, max_new_tokens=275)
    
    # Generate text using the instruction format
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result)  # Print the raw result for debugging
    
    # Return the generated text
    return result[0]['generated_text']

# Main function for drug recommendation with semantic key extraction
def get_drug_recomendation(age, sex, current_medication, symptoms, other_details):
    """
    Generate drug recommendations based on patient details using semantic key extraction.
    
    Args:
        age (str): Patient's age
        sex (str): Patient's sex
        current_medication (str): Medications the patient is currently taking
        symptoms (str): Patient's symptoms
        other_details (str): Additional relevant information
        
    Returns:
        str: Formatted drug recommendations
    """
    # Step 1: Extract relevant medical conditions using the smaller LLM
    # Define the prompt for the semantic key task
    small_llm_prompt = f"""
    You are a highly knowledgeable medical assistant. Based on the symptoms provided, identify the medical conditions that most closely relate to these symptoms.

    Age: {age}
    Sex: {sex}
    Symptoms: {symptoms}

    Your response can only include conditions which are present in the following list:
    {', '.join(condition_drug_dict.keys())}

    Provide the output in the format of a list of conditions, ranked by their relevance to the symptoms provided. The output can only include the ordered list with the following format, with no additional text afterwards or before:

    1.
    2.
    3.
    """

    # Get the output from the small LLM to identify conditions
    identified_conditions_text = prompt_tiny_model(small_llm_prompt)

    # Parse the output to extract the identified conditions
    identified_conditions = []
    for line in identified_conditions_text.split('\n'):
        line = line.strip()  # Remove leading and trailing whitespace
        if line and line[0].isdigit():  # Check if the line starts with a digit
            # Remove the leading number, period, and any spaces that follow
            condition = line.split('. ', 1)[-1].strip()
            identified_conditions.append(condition)

    print(identified_conditions)  # Print identified conditions for debugging

    # Step 2: Retrieve associated drugs for each identified condition
    recommended_medications = []
    for condition in identified_conditions[:6]:  # Consider top 6 conditions
        if condition in condition_drug_dict:
            # Get top 4 medications for each condition
            recommended_medications.extend(condition_drug_dict[condition][:4])

    # Step 3: Generate final recommendations using the fine-tuned LLM
    large_llm_prompt = f"""
    You are a highly knowledgeable medical assistant. Your task is to recommend medications based on the patient's details and symptoms. The output must **strictly follow the exact format below** without deviation.

    **Do not include any extra words, explanations, or greetings.** The response must contain only two sections:

    **1. Ranked List of Medications**:
    Provide a simple, numbered list of medication names in the following format:

    1. <Medication 1>
    2. <Medication 2>
    3. <Medication 3>

    **2. Brief Explanation**:
    After the medication list, provide a **brief and standardized explanation** for why these medications were chosen, including considerations such as drug interactions, patient age, sex, and specific symptoms. (2-3 sentences max)

    **Do not deviate from this format**.

    **Patient Details**:
    Age: {age}
    Sex: {sex}
    Current Medications: {current_medication}
    Symptoms: {symptoms}
    Other Relevant Details: {other_details}

    Here are medications to consider: {', '.join(recommended_medications)}
"""

    # Generate the final recommendation using the fine-tuned LLM
    return prompt_model(large_llm_prompt)

# %%
# Implementation 2: Direct recommendation without semantic key extraction
# This approach bypasses condition extraction and directly generates recommendations
# from patient symptoms using only the fine-tuned LLM

def get_drug_recomendation(age, sex, current_medication, symptoms, other_details):
    """
    Generate drug recommendations based on patient details without semantic key extraction.
    
    Args:
        age (str): Patient's age
        sex (str): Patient's sex
        current_medication (str): Medications the patient is currently taking
        symptoms (str): Patient's symptoms
        other_details (str): Additional relevant information
        
    Returns:
        str: Formatted drug recommendations
    """
    # Define the prompt for the fine-tuned LLM
    large_llm_prompt = f"""
    You are a highly knowledgeable medical assistant. Your task is to recommend medications based on the patient's details and symptoms. The output must **strictly follow the exact format below** without deviation.

    **Do not include any extra words, explanations, or greetings.** The response must contain only two sections:

    **1. Ranked List of Medications**:
    Provide a simple, numbered list of medication names in the following format:

    1. <Medication 1>
    2. <Medication 2>
    3. <Medication 3>

    **2. Brief Explanation**:
    After the medication list, provide a **brief and standardized explanation** for why these medications were chosen, including considerations such as drug interactions, patient age, sex, and specific symptoms. (2-3 sentences max)

    **Do not deviate from this format**.

    **Patient Details**:
    Age: {age}
    Sex: {sex}
    Current Medications: {current_medication}
    Symptoms: {symptoms}
    Other Relevant Details: {other_details}
"""

    # Generate recommendations directly using the fine-tuned LLM
    return prompt_model(large_llm_prompt)

# 
# # Automatic Testing
# This section automatically tests the recommendation system on a dataset
# of patient cases and records the outputs for later evaluation.

# %%
# Open the input CSV file containing test cases
inputs_file = open("/content/model_inputs_or.csv", "r")

# Create a CSV reader
reader = csv.reader(inputs_file)

# Skip the header row to get to the data
next(reader)

# Open a new CSV file to write the results
with open("/content/lamma2_non_finetuned_model_outputs.csv", "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write the header row for the output file
    csvwriter.writerow(["age", "sex", "current_medication", "symptoms", "other_details", "drs_output", "reviwed_medication", "review_score"])

    # Process each test case
    for row in reader:
        # Check if the row has the expected number of elements (7)
        if len(row) == 7:
            # Unpack the row into individual variables
            age, sex, current_medication, symptoms, other_details, reviewed_medication, review_rating = row

            # Generate a recommendation for this patient case
            response = get_drug_recomendation(age, sex, current_medication, symptoms, other_details)
            
            # Write the input data, generated recommendation, and ground truth to the output file
            csvwriter.writerow([age, sex, current_medication, symptoms, other_details, response, reviewed_medication, review_rating])
        else:
            # Handle rows with unexpected number of elements by logging an error
            print(f"Skipping row due to unexpected number of elements: {row}")

        # This block appears to be duplicated from above and may be an error in the original code
        # It generates recommendations again and writes them to the output file
        response = get_drug_recomendation(age, sex, current_medication, symptoms, other_details)
        csvwriter.writerow([age, sex, current_medication, symptoms, other_details, response, reviewed_medication, review_rating])