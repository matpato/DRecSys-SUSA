# 
# ### Data Visualization for the Evaluation of the DRecSys (Drug Recommendation System)
# 
# This script performs visualization and evaluation of the Drug Recommendation System's output.
# It analyzes how well the system's medication suggestions align with medications that users actually
# rated positively.
#
# Organization of the file:
# 
# 1. Imports - Libraries needed for data processing and visualization
# 2. Read the dataset - Load model outputs from CSV files
# 3. LLM Medication Guess Performance - Analyze how well the model recommends medications
# 4. Ranking Metrics - Calculate standard information retrieval metrics
# 5. Extraction of information - Analyze most commonly suggested medications
# 

# %%
# Code 1: Imports and Style Configuration 

import pandas as pd      # For data manipulation and analysis
import re                # For regular expressions (extracting medications from text)
import matplotlib.pyplot as plt  # For creating visualizations
import numpy as np       # For numerical operations (used in metrics calculation)

# Set the style of our plots to a more visually appealing grid style
plt.style.use('ggplot')

# %%
# Code 2: Read the Model Output CSV file
# This section loads the results from a previously run recommendation generation.
# Multiple file paths are provided but commented out - uncomment the one needed.

# Available results files:
# file_path_new = r'output/model_outputs_or.csv'                     # Original ratings model
# file_path_new = r'output/model_outputs_or_no_semantic.csv'         # Original ratings without semantic extraction
# file_path_new = r'output/model_outputs_clean_vader_no_semantic.csv' # Clean VADER without semantic
# file_path_new = r'output/model_outputs_clean_vader.csv'            # Clean VADER with semantic
# file_path_new = r'output/model_outputs_raw_vader_no_semantic.csv'  # Raw VADER without semantic

# Currently selected file (non-fine-tuned model outputs)
file_path_new = r'output/lamma2_non_finetuned_model_outputs.csv'

# Read the CSV file into a pandas DataFrame
df_new = pd.read_csv(file_path_new)

# %%
# Code 3: LLM Medication Recommendation Performance Analysis
# This section evaluates how well the model's recommendations match medications that users rated highly.

# Function to extract the ranked medications from the LLM output text
def clean_medication_list(output):
    """
    Extract medication names from the LLM's output text using regex.
    
    Args:
        output (str): Raw text output from the LLM
        
    Returns:
        list: List of extracted medication names, or ['Extraction Failed'] if none found
    """
    # Regex pattern to find text after numbered list items (e.g., "1. Medication")
    # (?<=\d\.\s) is a positive lookbehind for a digit, period, and space
    # ([A-Za-z\(\)\-]+) captures one or more letters, parentheses, or hyphens after that
    pattern = r'(?<=\d\.\s)([A-Za-z\(\)\-]+)'
    
    # Find all matches in the output text
    medications = re.findall(pattern, output)
    
    # Filter out any irrelevant words that might have been captured
    irrelevant_words = {"Ranked", "Brief", "Explanation", "Medication"}
    medications = [med for med in medications if med not in irrelevant_words]
    
    # Return the list of medications, or ['Extraction Failed'] if empty
    return medications if medications else ['Extraction Failed']

# Define function to categorize each recommendation based on ranking and review score
def categorize_guesses_shorter(row):
    """
    Categorize the recommendation quality based on whether the reviewed medication
    was recommended and what rating the user gave it.
    
    Args:
        row: DataFrame row containing recommendation and review data
        
    Returns:
        str: Category label describing the recommendation quality
    """
    # Get the list of recommended medications and the actual reviewed medication
    ranked_meds = row['ranked_medications']
    reviewed_med = row['reviwed_medication']  # Note: typo in column name is from the original data
    
    # Initialize score and handle potential conversion issues
    score = 0
    try:
        score = float(row['review_score'])
    except ValueError:
        pass  # If conversion fails, keep score as 0
    except KeyError:
        pass  # If the key doesn't exist, keep score as 0
             
    # Check if the model provided valid recommendations
    if ranked_meds == ['Extraction Failed']:
        return 'Extraction Failed'
    
    # Check if the reviewed medication is in the recommended list
    if reviewed_med in ranked_meds:
        # Get the 1-based rank position of the reviewed medication
        rank = ranked_meds.index(reviewed_med) + 1
        
        # Categorize based on user rating (score) and rank
        if score >= 7:  # User liked the medication (high rating)
            if rank == 1:
                return 'User liked\ntop suggestion'  # True Positive, top recommendation
            else:
                return 'User liked\nother\nsuggestions'  # True Positive, but not top recommendation
        elif score >= 5:  # User was neutral about the medication
            return 'User was\nneutral to\na suggestion'  # Somewhat positive
        else:  # User disliked the medication (low rating)
            return 'User Disliked'  # False Positive - system recommended something user didn't like
    else:
        # The reviewed medication wasn't in the recommendations
        if score >= 7:
            return 'Inconclusive\n(Positive)'  # False Negative - user liked but system didn't recommend
        elif score >= 5:
            return 'Inconclusive\n(Neutral)'   # User was neutral, system didn't recommend
        else:
            return 'Not liked by user \n& not suggested'  # True Negative - correctly didn't recommend

# Apply the extraction function to parse medication names from the model output
df_new['ranked_medications'] = df_new['drs_output'].apply(clean_medication_list)

# Apply the categorization function to evaluate each recommendation
df_new['guess_category'] = df_new.apply(categorize_guesses_shorter, axis=1)

# Count the occurrences of each category
guess_summary_shorter = df_new['guess_category'].value_counts()

# Define color mapping for each category to make the visualization more intuitive
# Green: Good outcomes, Yellow: Inconclusive, Red: Bad outcomes, Gray: Errors
color_mapping = {
    'User liked\ntop suggestion': 'tab:green',           # Success - top recommendation
    'User liked\nother\nsuggestions': 'tab:green',       # Success - other positions
    'Not liked by user \n& not suggested': 'tab:green',  # Success - correctly didn't recommend
    'Inconclusive\n(Positive)': '#DAA520',               # Mixed - user liked but not recommended
    'Inconclusive\n(Neutral)': '#DAA520',                # Mixed - neutral, not recommended
    'User was\nneutral to\na suggestion': 'tab:green',   # Somewhat positive
    'Extraction Failed': 'gray',                         # Error in processing
    'User Disliked': 'tab:red'                           # Failure - recommended something disliked
}

# Map colors to each category in the results
colors = [color_mapping.get(category, 'gray') for category in guess_summary_shorter.index]

# Create a bar chart to visualize the distribution of categories
plt.figure(figsize=(10, 6))
guess_summary_shorter.plot(kind='bar', color=colors, width=0.4, figsize=(10, 5))
plt.title('LLM Medication Guess Performance')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Keep category labels horizontal for readability

# Add a legend to explain the color coding
legend_labels = ['Green = Positive', 'Yellow = Inconclusive', 'Red = Negative', 'gray = Error']
plt.legend(handles=[
    plt.Line2D([0], [0], color='tab:green', lw=4, label=legend_labels[0]),
    plt.Line2D([0], [0], color='#DAA520', lw=4, label=legend_labels[1]),
    plt.Line2D([0], [0], color='tab:red', lw=4, label=legend_labels[2]),
    plt.Line2D([0], [0], color='gray', lw=4, label=legend_labels[3])
],
title="Legend")

plt.tight_layout()
plt.show()

# Print the exact counts for each category
print(guess_summary_shorter)



# %%
# Code: Calculate Standard Information Retrieval Metrics
# This section calculates standard metrics used in recommendation system evaluation:
# - Hit Rate (HR): Fraction of cases where the top recommendation matches the user's preferred item
# - Mean Reciprocal Rank (MRR): Average of 1/rank of the first relevant item
# - Normalized Discounted Cumulative Gain (NDCG): Measures ranking quality with position discounting

# Re-apply the extraction function to ensure we have the ranked medications
# (This is redundant with the earlier extraction but kept for script independence)
df_new['ranked_medications'] = df_new['drs_output'].apply(clean_medication_list)

# Define a function to calculate evaluation metrics for each row
def calculate_metrics(row):
    """
    Calculate recommendation quality metrics for a single case.
    
    Args:
        row: DataFrame row containing recommendation and review data
        
    Returns:
        pandas.Series: Hit rate, reciprocal rank, and NDCG metrics
    """
    # Get the recommended medications and the one actually reviewed
    ranked_meds = row['ranked_medications']
    reviewed_med = row['reviwed_medication']
    
    # Initialize score and handle potential conversion issues
    score = 0
    try:
        score = float(row['review_score'])
    except ValueError:
        pass
    except KeyError:
        pass  
    
    # Hit Rate: Binary metric - 1 if top recommendation matches reviewed medication
    # Only consider positive reviews (score >= 7) and valid extractions
    hit = ranked_meds[0] == reviewed_med if score >= 7 and ranked_meds != ['Extraction Failed'] else 0
    
    # Reciprocal Rank: 1/position of the relevant item
    # Only consider positive reviews (score >= 7)
    if reviewed_med in ranked_meds and score >= 7:
        rank = ranked_meds.index(reviewed_med) + 1  # 1-based index
        reciprocal_rank = 1 / rank
    else:
        reciprocal_rank = 0
    
    # Discounted Cumulative Gain (DCG): Sum of gains discounted by position
    # Normalized DCG (NDCG): DCG divided by ideal DCG
    dcg = 0
    idcg = 0
    
    # Calculate DCG - sum of relevance scores discounted by log2(position+1)
    for idx, med in enumerate(ranked_meds[:10]):  # Consider top 10 recommendations
        # Binary relevance: 1 if medication matches and was liked, 0 otherwise
        relevance = 1 if med == reviewed_med and score >= 7 else 0
        # Apply logarithmic discount based on position
        dcg += relevance / np.log2(idx + 2)  # +2 because idx is 0-based and log2(1) would be 0
        
    # Ideal DCG - what would DCG be if the relevant item was ranked first
    # Only applies if the user liked the medication (score >= 7)
    idcg = 1 / np.log2(2) if score >= 7 else 0
    
    # Calculate NDCG - normalize DCG against the ideal case
    # This makes NDCG range from 0 (worst) to 1 (perfect ranking)
    ndcg = dcg / idcg if idcg > 0 else 0

    # Return all metrics as a pandas Series
    return pd.Series({'hit': hit, 'reciprocal_rank': reciprocal_rank, 'ndcg': ndcg})

# Apply the metrics calculation to each row in the DataFrame
df_new[['hit', 'reciprocal_rank', 'ndcg']] = df_new.apply(calculate_metrics, axis=1)

# Calculate aggregate metrics across all cases
hit_rate = df_new['hit'].mean()       # Average hit rate
mrr = df_new['reciprocal_rank'].mean() # Mean reciprocal rank
ndcg = df_new['ndcg'].mean()          # Average NDCG

# Print the aggregate metrics with 4 decimal places
print(f"Hit Rate (HR): {hit_rate:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"Normalized Discounted Cumulative Gain (NDCG): {ndcg:.4f}")


# %%
# Code 4: Extract Most Frequently Recommended Medications
# This section analyzes which medications are most commonly suggested by the model

# Flatten the list of ranked medications into a single list
all_ranked_medications = df_new['ranked_medications'].explode()

# Count the frequency of each medication in the recommendations
ranked_medication_counts = all_ranked_medications.value_counts()

# Get the top 10 most frequently suggested medications
top_10_suggested_medications = ranked_medication_counts.head(10)

# Display the top 10 most suggested medications with their frequencies
print("Top 10 most suggested medications (including all entries):")
print(top_10_suggested_medications)


# %%
# Code 4: Extract Most Frequently Recommended Medications (with filtering)
# This is a refined version that filters out irrelevant entries

# Re-flatten the list of ranked medications
all_ranked_medications = df_new['ranked_medications'].explode()

# Count the frequency of each medication again
ranked_medication_counts = all_ranked_medications.value_counts()

# Filter out entries that are likely incorrect or irrelevant:
# 1. Remove entries containing "The" which might be fragments of sentences
# 2. Remove "Extraction Failed" entries which indicate parsing errors
filtered_medications = ranked_medication_counts[
    ~ranked_medication_counts.index.str.contains(r'\bThe\b', case=False) &
    (ranked_medication_counts.index != 'Extraction Failed')
]

# Get the top 10 most frequently suggested medications after filtering
top_10_suggested_medications = filtered_medications.head(10)

# Display the filtered top 10 most suggested medications
print("\nTop 10 most suggested medications (filtered):")
print(top_10_suggested_medications)



# %%
# Code: Alternative Visualization of LLM Medication Recommendation Performance
# This section provides another way to visualize performance with simplified categories

# Define an alternative categorization function with different grouping
def categorize_guesses_shorter(row):
    """
    Alternative categorization function with different category definitions.
    This version uses a simpler set of categories compared to the previous one.
    
    Args:
        row: DataFrame row containing recommendation and review data
        
    Returns:
        str: Category label for the case
    """
    ranked_meds = row['ranked_medications']
    reviewed_med = row['reviwed_medication']
    score = row['review_score']
    
    # Check for extraction failures
    if ranked_meds == ['Extraction Failed']:
        return 'Extraction Failed'
    
    # Check if the reviewed medication is in the recommended list
    if reviewed_med in ranked_meds:
        rank = ranked_meds.index(reviewed_med) + 1  # Get 1-based index
        if score >= 7:  # User liked the medication
            if rank == 1:
                return 'User liked\ntop suggestion' 
            else:
                return 'User liked\nother suggestions'
        elif score >= 5:  # User was neutral
            return 'Inconclusive\nNeutral' 
        else:  # User disliked
            return 'User Disliked'
    else:
        # Medication wasn't recommended
        if score >= 7:  # But user liked it
            return 'Inconclusive'  # False negative
        elif score >= 5:  # User was neutral
            return 'Inconclusive'  # Less concerning false negative
        else:  # User didn't like it
            return 'Not user \npreferred \nnot recommended'  # True negative

# Re-extract medications (redundant but kept for section independence)
df_new['ranked_medications'] = df_new['drs_output'].apply(clean_medication_list)

# Apply the alternative categorization
df_new['guess_category'] = df_new.apply(categorize_guesses_shorter, axis=1)

# Count the occurrences of each category
guess_summary_shorter = df_new['guess_category'].value_counts()

# Create a bar chart with all categories colored green
# This is a simpler visualization without the category-specific coloring
plt.figure(figsize=(10, 6))
guess_summary_shorter.plot(kind='bar', color='tab:green', width=0.4, figsize=(10, 5))
plt.title('LLM Medication Guess Performance')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Keep labels horizontal
plt.tight_layout()
plt.show()