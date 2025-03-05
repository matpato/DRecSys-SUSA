# ### Dataset Construction
# 
# This script handles the construction of datasets for the Drug Recommendation System (DRecSys).
# It processes raw drug review data into formats suitable for training and evaluation.
#
# Organization of the file:
# 1. Imports - Libraries and modules needed for data processing
# 2. Read the dataset - Download and load CSV files from GitHub repository
# 3. Data cleaning - Remove corrupted rows from the original dataset
# 4. Dataset creation for ranking - Create condition-to-drug mappings sorted by ratings
# 5. Dataset creation for fine-tuning - Format data for LLM fine-tuning

# %%
# Code 1: Imports
# Essential libraries for data processing, HTTP requests, and file operations

import requests          # For HTTP requests to download data from GitHub
import pandas as pd      # For data manipulation and analysis
import os                # For file system operations
from urllib.parse import urljoin  # For URL manipulation
import json              # For parsing JSON responses from GitHub API

# %%
# Code 2: Read the dataset
# This function downloads all CSV files from a specified GitHub folder

def download_github_folder_csvs(username, repository, branch, folder_path, local_folder="downloaded_csvs"):
    """
    Download all CSV files from a GitHub folder
    
    Parameters:
    username (str): GitHub username
    repository (str): Repository name
    branch (str): Branch name (usually 'main' or 'master')
    folder_path (str): Path to the folder within the repository
    local_folder (str): Local folder to save the downloaded files
    
    Returns:
    list: List of dictionaries containing file info and dataframes
    """
    # Create local folder if it doesn't exist
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    
    # Construct GitHub API URL to get contents of a folder
    api_url = f"https://api.github.com/repos/{username}/{repository}/contents/{folder_path}?ref={branch}"
    
    # Send GET request to GitHub API
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"Error accessing GitHub API: {response.status_code}")
        return
    
    # Parse JSON response to get folder contents
    contents = json.loads(response.text)
    
    # Filter for CSV files and download each one
    downloaded_files = []
    for item in contents:
        if item["type"] == "file" and item["name"].endswith(".csv"):
            # Get raw file URL
            raw_url = item["download_url"]
            
            # Download file directly into pandas DataFrame
            print(f"Downloading {item['name']}...")
            df = pd.read_csv(raw_url)
            
            # Save to local file for caching
            local_path = os.path.join(local_folder, item["name"])
            df.to_csv(local_path, index=False)
            
            # Store file info and DataFrame in result list
            downloaded_files.append({
                "name": item["name"],
                "path": local_path,
                "dataframe": df
            })
            print(f"Saved to {local_path}")
    
    print(f"Downloaded {len(downloaded_files)} CSV files")
    return downloaded_files

# Repository information
username = "matpato"
repository = "EDRISA"
branch = "main"  # or 'master' depending on your repository
folder_path = "/data"  # Path to the folder containing CSV files

# Download all CSV files from the GitHub repository
datasets = download_github_folder_csvs(username, repository, branch, folder_path)

# Display information about the downloaded datasets
if datasets:
    for dataset in datasets:
        print(f"File: {dataset['name']}, Shape: {dataset['dataframe'].shape}")

# Select the second dataset for processing (assuming it's the drug reviews dataset)
ds = dataset[1]

# %%
# Code 3: Data cleaning of the original dataset
# Remove corrupted rows that don't have proper condition information

# Identify corrupted reviews where the 'condition' field contains text that should be elsewhere
# This pattern suggests a parsing error in the original data collection
num_corrupted_reviews = len(ds[ds['condition'].str.contains("users found this comment helpful.", na=False)])
print("Number of Corrupted Reviews: ", num_corrupted_reviews)

# Remove the corrupted rows from the dataset
# ~ds['condition']... means "rows where condition does NOT contain the pattern"
ds = ds[~ds['condition'].str.contains(" users found this comment helpful.", na=False)]

# %%
# Code 4: Dataset creation for ranking (Original rating-based)
# Create a mapping of medical conditions to ranked drugs based on original user ratings

# Sort the dataset by condition (alphabetically) and rating (highest first)
ds_sorted = ds.sort_values(by=['condition', 'rating'], ascending=[True, False])

# Group by 'condition' and aggregate the 'drugName' into a list
# This creates a mapping where each condition has an associated list of drugs, ordered by rating
result_ds = ds_sorted.groupby('condition')['drugName'].agg(list).reset_index()

# Rename columns to reflect the data they contain
result_ds.columns = ['Condition', 'Drugs']

# Save the result to a new CSV file for later use in the recommendation system
result_ds.to_csv(r'output/condition_drugs_or.csv', index=False)
print(result_ds.head())  # Display the first few rows of the new DataFrame

# %%
# Code 4.1: Dataset creation for ranking (VADER sentiment-based)
# Create a mapping of conditions to drugs ranked by VADER sentiment scores
# VADER is a sentiment analysis tool that may provide more nuanced ranking than raw ratings

# Sort the dataset by condition and VADER sentiment score (highest first)
# VADER_rescaled is presumed to be a sentiment score derived from the review text
ds_sorted = ds.sort_values(by=['condition', 'VADER_rescaled'], ascending=[True, False])

# Group by 'condition' and aggregate the 'drugName' into a list
result_ds = ds_sorted.groupby('condition')['drugName'].agg(list).reset_index()

# Rename columns to reflect the data they contain
result_ds.columns = ['Condition', 'Drugs']

# Save the result to a new CSV file for the VADER-based recommendation approach
result_ds.to_csv(r'output/condition_drugs_vader.csv', index=False)
print(result_ds.head())  # Display the first few rows of the new DataFrame

# %%
# Code 5: Dataset creation for fine-tuning the LLM (Original rating-based)
# Format the data for fine-tuning with original ratings

# Define a function to create formatted input and output text for the LLM
def format_rows(row):
    """
    Format each row into prompt-completion pairs for LLM fine-tuning
    
    Parameters:
    row: DataFrame row containing drug review data
    
    Returns:
    Series: Two columns - 'prompt' for LLM input and 'completion' for expected output
    """
    # Format input text (prompt) describing the review scenario
    input_text = f"User review on the medication {row['drugName']} for {row['condition']}"
    
    # Format output text (completion) with rating and review details
    # ascii() ensures the text is properly escaped for training data
    output_text = ascii(f"User gave it a rating of {row['rating']}:\n{row['review']}. {row['usefulCount']} people found this review useful.").strip()
    
    return pd.Series([input_text, output_text], index=['prompt', 'completion'])

# Apply the formatting function to create a new dataset for fine-tuning
new_ds = ds.apply(format_rows, axis=1)

# Save the formatted dataset
# Uncomment the appropriate line depending on whether this is for training or testing
#new_ds.to_csv(r'output/train_ready_fine_tuning_or.csv', index=False)
new_ds.to_csv(r'output/test_ready_fine_tuning_or.csv', index=False)

print(new_ds.head())  # Display the first few rows of the new DataFrame

# %%
# Code 5.1: Dataset creation for fine-tuning the LLM (VADER sentiment-based)
# Format the data for fine-tuning using VADER sentiment scores instead of original ratings

# Define a function to create input and output columns with VADER scores
def format_rows(row):
    """
    Format each row into prompt-completion pairs using VADER sentiment scores
    
    Parameters:
    row: DataFrame row containing drug review data
    
    Returns:
    Series: Two columns - 'prompt' for LLM input and 'completion' for expected output
    """
    # Format input text (prompt) describing the review scenario
    input_text = f"User review on the medication {row['drugName']} for {row['condition']}"
    
    # Format output text (completion) with VADER sentiment score and review details
    output_text = ascii(f"User gave it a rating of {row['VADER_rescaled']}:\n{row['review']}. {row['usefulCount']} people found this review useful.").strip()
    
    return pd.Series([input_text, output_text], index=['prompt', 'completion'])

# Apply the formatting function to create a new dataset for VADER-based fine-tuning
new_ds = ds.apply(format_rows, axis=1)

# Save the formatted dataset - uncomment the appropriate line
# Different versions are created for different experiments:
# - clean_vader: Presumably cleaned or processed VADER scores
# - raw_vader: Raw VADER sentiment scores

#new_ds.to_csv(r'output/train_ready_fine_tuning_clean_vader.csv', index=False)
#new_ds.to_csv(r'output/test_ready_fine_tuning_clean_vader.csv', index=False)

#new_ds.to_csv(r'output/train_ready_fine_tuning_raw_vader.csv', index=False)
new_ds.to_csv(r'output/test_ready_fine_tuning_raw_vader.csv', index=False)

print(new_ds.head())  # Display the first few rows of the new DataFrame