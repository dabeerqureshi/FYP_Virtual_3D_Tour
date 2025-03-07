import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

# Function to calculate the cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Function to generate Bayesian Inference-based prompt
def generate_bayesian_prompt(row):
    # Convert embedding strings to lists
    top_embeddings = [
        eval(row['Top1_Embedding']),
        eval(row['Top2_Embedding']),
        eval(row['Top3_Embedding']),
        eval(row['Top4_Embedding']),
        eval(row['Top5_Embedding']),
    ]
    top_labels = [
        row['Top1_Label'],
        row['Top2_Label'],
        row['Top3_Label'],
        row['Top4_Label'],
        row['Top5_Label'],
    ]
    
    # Prior probabilities (assuming equal priors for simplicity)
    prior_probs = {label: 1/5 for label in top_labels}
    
    # Calculate likelihoods using cosine similarity
    likelihoods = {}
    for label, embedding in zip(top_labels, top_embeddings):
        # Compare the Top1 embedding with others
        likelihood = cosine_similarity(eval(row['Top1_Embedding']), embedding)
        likelihoods[label] = likelihood

    # Normalize likelihoods (sum of likelihoods = 1)
    total_likelihood = sum(likelihoods.values())
    for label in likelihoods:
        likelihoods[label] /= total_likelihood

    # Compute the posterior probabilities using Bayes' Theorem: P(label|embedding) ∝ P(embedding|label) * P(label)
    posterior_probs = {}
    for label in likelihoods:
        posterior_probs[label] = likelihoods[label] * prior_probs[label]
    
    # Normalize posterior probabilities
    total_posterior = sum(posterior_probs.values())
    for label in posterior_probs:
        posterior_probs[label] /= total_posterior

    # Generate the prompt based on the posterior probabilities
    sorted_labels = sorted(posterior_probs, key=posterior_probs.get, reverse=True)
    top_label = sorted_labels[0]
    prompt = f"Based on the Bayesian Inference, the image is most likely a {top_label} with a probability of {posterior_probs[top_label]:.2f}. The other possibilities are: {', '.join([f'{label} ({posterior_probs[label]:.2f})' for label in sorted_labels[1:]])}."
    return prompt

def generate_prompts_from_csv(input_file, output_file='generated_prompts.csv'):
    # Load the data into a pandas DataFrame
    df = pd.read_csv(input_file)
    
    # Apply the prompt generation function to each row
    df['Prompt'] = df.apply(generate_bayesian_prompt, axis=1)
    
    # Save the DataFrame with the generated prompts to a CSV file
    df[['Image Path', 'Prompt']].to_csv(output_file, index=False)
    print(f"Prompts have been saved to {output_file}")

# Example usage
input_file = r"C:\Users\qures\OneDrive\Desktop\Generating_Dataset\example.csv"  # Replace with your input file path
output_file = r'C:\Users\qures\OneDrive\Desktop\Generating_Dataset\generated_prompts.csv'  # Path to save the output
generate_prompts_from_csv(input_file, output_file)
