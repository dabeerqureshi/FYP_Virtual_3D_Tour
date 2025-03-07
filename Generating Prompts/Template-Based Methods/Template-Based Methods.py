import pandas as pd

# Function to generate prompts using templates
def generate_prompt_template(row):
    # Example of template: "This image represents a [Top1_Label] with features [Top1_Embedding]."
    label = row['Top1_Label']
    embedding = row['Top1_Embedding']  # Assuming the embedding is in a string format like "[0.12, 0.45, 0.78]"
    
    # Converting the string of the embedding into a list (if necessary)
    embedding_list = eval(embedding)  # Safe if the embeddings are always valid Python lists
    features = ', '.join([f"{feature:.2f}" for feature in embedding_list])  # Format features for readability
    
    # Fill the template
    prompt = f"This image features a {label} with features {features}."
    
    return prompt

# Function to process the dataset and generate prompts
def generate_prompts(input_file, output_file='generated_prompts_template.csv'):
    # Load the data into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Check if required columns exist
    print("Columns available:", df.columns)
    
    # Apply the template-based prompt generation function to each row
    df['Prompt'] = df.apply(generate_prompt_template, axis=1)

    # Save the DataFrame with the generated prompts to a CSV file
    df[['Image Path', 'Prompt']].to_csv(output_file, index=False)
    print(f"Prompts have been saved to {output_file}")

# Example usage
input_file = r"C:\Users\qures\OneDrive\Desktop\Generating_Dataset\example.csv"  # Replace with your input file path
output_file = r'C:\Users\qures\OneDrive\Desktop\Generating_Dataset\generated_prompts_template.csv'  # Path to save the output
generate_prompts(input_file, output_file)
