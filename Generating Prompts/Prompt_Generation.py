import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the LLM model and tokenizer
model_name = "t5-small"  # Replace with a larger model like T5-base if needed
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate a prompt using labels
def generate_prompt(labels):
    # Construct input text for the LLM
    input_text = f"Describe an image containing: {', '.join(labels)}."
    
    # Generate output from the LLM
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=64, num_beams=4)
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the dataset
input_file = r"C:\Users\qures\OneDrive\Desktop\FYP\Generating Prompts\10_Images_Example.csv"  # Replace with your dataset's path
df = pd.read_csv(input_file)

# Prepare for storing generated prompts
prompts = []

for _, row in df.iterrows():
    # Extract labels
    labels = [row[f"Top{i}_Label"] for i in range(1, 6) if pd.notna(row[f"Top{i}_Label"])]
    
    # Generate prompt using the labels
    prompt = generate_prompt(labels)
    prompts.append({"Image Path": row["Image Path"], "Generated Prompt": prompt})

# Save the results to a new CSV
output_df = pd.DataFrame(prompts)
output_file = r"C:\Users\qures\OneDrive\Desktop\FYP\Generating Prompts\generated_prompts_10.csv"
output_df.to_csv(output_file, index=False)

print(f"Generated prompts saved to {output_file}")
