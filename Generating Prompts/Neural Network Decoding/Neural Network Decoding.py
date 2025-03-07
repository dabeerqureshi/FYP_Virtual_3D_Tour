import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# Example of a simple neural network model for embedding classification
class EmbeddingDecoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingDecoderModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Dataset class to handle embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Function to decode embeddings using a trained neural network model
def generate_prompt_with_neural_network(row, model, label_encoder):
    embedding = np.array(eval(row['Top1_Embedding']))  # Convert string to list of floats
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    output = model(embedding_tensor)
    _, predicted_class = torch.max(output, 1)
    label = label_encoder.inverse_transform([predicted_class.item()])[0]
    prompt = f"Based on the neural network model, this image most likely represents {label}, which is identified by the model from the features in the image."
    return prompt

# Train a neural network (or load a pre-trained model)
def train_model(embeddings, labels, input_dim, hidden_dim, output_dim, epochs=10, batch_size=32, learning_rate=0.001):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    dataset = EmbeddingDataset(embeddings, encoded_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EmbeddingDecoderModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    return model, label_encoder

# Main function to generate prompts
def generate_prompts_with_neural_network(input_file, output_file='generated_prompts_nn.csv'):
    # Load the data into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Check if the required columns exist
    print("Columns available:", df.columns)
    
    # Extract embeddings and labels for training or inference
    embeddings = []
    labels = []
    for i in range(1, 6):
        embeddings_col = f'Top{i}_Embedding'
        if embeddings_col in df.columns:
            embeddings.extend([eval(embedding) for embedding in df[embeddings_col]])
            labels.extend(df[f'Top{i}_Label'].values)
        else:
            print(f"Column {embeddings_col} not found!")

    # Flatten embeddings for clustering
    embeddings = np.array(embeddings).reshape(-1, len(eval(df['Top1_Embedding'][0])))  # Assuming embedding length is consistent

    # Model and label training (or load pre-trained model)
    input_dim = len(embeddings[0])  # The size of the embedding vector
    hidden_dim = 128  # Hidden layer size
    output_dim = len(set(labels))  # Number of unique labels

    model, label_encoder = train_model(embeddings, labels, input_dim, hidden_dim, output_dim, epochs=10)

    # Apply the prompt generation function to each row in the DataFrame
    df['Prompt'] = df.apply(generate_prompt_with_neural_network, axis=1, model=model, label_encoder=label_encoder)

    # Save the DataFrame with the generated prompts to a CSV file
    df[['Image Path', 'Prompt']].to_csv(output_file, index=False)
    print(f"Prompts have been saved to {output_file}")

# Example usage
input_file = r"C:\Users\qures\OneDrive\Desktop\Generating_Dataset\example.csv"  # Replace with your input file path
output_file = r'C:\Users\qures\OneDrive\Desktop\Generating_Dataset\generated_prompts_nn.csv'  # Path to save the output
generate_prompts_with_neural_network(input_file, output_file)
