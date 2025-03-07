import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Function to perform clustering and generate prompts
def generate_centroid_based_prompt(row, kmeans_model, embeddings_columns):
    embedding = np.array(eval(row[embeddings_columns]))  # Convert string to list of floats
    cluster = kmeans_model.predict([embedding])[0]
    centroid = kmeans_model.cluster_centers_[cluster]
    prompt = f"Based on the clustering analysis, this image most likely belongs to the cluster with the centroid at {centroid[:3]}. The image features resemble objects in this cluster, such as {row['Top1_Label']}, which are common in this cluster."
    return prompt

def generate_prompts_with_clustering(input_file, output_file='generated_prompts_cluster.csv', n_clusters=3):
    # Load the data into a pandas DataFrame
    df = pd.read_csv(input_file)
    
    # Print the first few rows to check data
    print(df.head())
    
    # Check if the required columns exist
    print("Columns available:", df.columns)
    
    # Extract embeddings for clustering (Top1_Embedding to Top5_Embedding)
    embeddings = []
    for i in range(1, 6):
        embeddings_col = f'Top{i}_Embedding'
        if embeddings_col in df.columns:
            embeddings.append([eval(embedding) for embedding in df[embeddings_col]])
        else:
            print(f"Column {embeddings_col} not found!")
    
    # Flatten the list of embeddings into a 2D array for clustering
    embeddings = np.array(embeddings).reshape(-1, len(eval(df['Top1_Embedding'][0])))  # Assuming embedding length is consistent
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    
    # Apply the prompt generation function to each row in the DataFrame
    df['Prompt'] = df.apply(generate_centroid_based_prompt, axis=1, kmeans_model=kmeans, embeddings_columns='Top1_Embedding')
    
    # Save the DataFrame with the generated prompts to a CSV file
    df[['Image Path', 'Prompt']].to_csv(output_file, index=False)
    print(f"Prompts have been saved to {output_file}")
    
# Example usage
input_file = r"C:\Users\qures\OneDrive\Desktop\Generating_Dataset\example.csv"  # Replace with your input file path
output_file = r'C:\Users\qures\OneDrive\Desktop\Generating_Dataset\generated_prompts_cluster.csv'  # Path to save the output
generate_prompts_with_clustering(input_file, output_file, n_clusters=3)
