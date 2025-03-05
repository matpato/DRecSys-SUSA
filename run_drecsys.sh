#!/bin/bash
set -e

# Print banner
echo "======================================================"
echo "  DRecSys: Drug Recommendation System Pipeline"
echo "======================================================"

# Create necessary directories
mkdir -p output
mkdir -p .cache/huggingface
mkdir -p .cache/transformers

# Check for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN environment variable not set."
  echo "You may need to set this to access gated models like Llama."
else
  # Log in to Hugging Face
  echo "Logging in to Hugging Face..."
  python -c "from huggingface_hub import login; login('$HF_TOKEN')"
fi

# Step 1: Process datasets
echo ""
echo "Step 1/4: Processing datasets..."
python datasets.py
if [ $? -ne 0 ]; then
  echo "Error processing datasets. Exiting."
  exit 1
fi
echo "✓ Datasets processed successfully."

# Step 2: Fine-tune the model (if requested)
if [ "$SKIP_FINETUNE" != "true" ]; then
  echo ""
  echo "Step 2/4: Fine-tuning model..."
  python finetune.py
  if [ $? -ne 0 ]; then
    echo "Error during fine-tuning. Exiting."
    exit 1
  fi
  echo "✓ Model fine-tuning completed."
else
  echo ""
  echo "Step 2/4: Fine-tuning skipped (SKIP_FINETUNE=true)."
fi

# Step 3: Generate recommendations
echo ""
echo "Step 3/4: Generating drug recommendations..."
python results.py
if [ $? -ne 0 ]; then
  echo "Error generating recommendations. Exiting."
  exit 1
fi
echo "✓ Recommendations generated successfully."

# Step 4: Evaluate results
echo ""
echo "Step 4/4: Evaluating recommendation performance..."
python evaluation.py
if [ $? -ne 0 ]; then
  echo "Error evaluating recommendations. Exiting."
  exit 1
fi
echo "✓ Evaluation completed."

# Summarize
echo ""
echo "======================================================"
echo "  DRecSys Pipeline Completed Successfully"
echo "======================================================"
echo "Output files are available in the 'output/' directory."
echo ""