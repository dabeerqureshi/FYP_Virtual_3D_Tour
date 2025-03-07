import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Example DataFrames for each method
data_bayesian = {
    "Image Path": ["img1.jpg", "img2.jpg"],
    "Prompt": [
        "Based on the Bayesian Inference, the image is most likely a mountain with a probability of 0.25. The other possibilities are: lake (0.24), valley (0.19), forest (0.17), river (0.16).",
        "Based on the Bayesian Inference, the image is most likely a monument with a probability of 0.25. The other possibilities are: plaza (0.21), temple (0.21), castle (0.17), statue (0.15)."
    ]
}

data_clustering = {
    "Image Path": ["img1.jpg", "img2.jpg"],
    "Prompt": [
        "Based on the clustering analysis, this image most likely belongs to the cluster with the centroid at [0.12 0.45 0.78]. The image features resemble objects in this cluster, such as mountain, which are common in this cluster.",
        "Based on the clustering analysis, this image most likely belongs to the cluster with the centroid at [0.6425 0.5875 0.3675]. The image features resemble objects in this cluster, such as monument, which are common in this cluster."
    ]
}

data_neural = {
    "Image Path": ["img1.jpg", "img2.jpg"],
    "Prompt": [
        "Based on the neural network model, this image most likely represents statue, which is identified by the model from the features in the image.",
        "Based on the neural network model, this image most likely represents forest, which is identified by the model from the features in the image."
    ]
}

data_template = {
    "Image Path": ["img1.jpg", "img2.jpg"],
    "Prompt": [
        "This image features a mountain with features 0.12, 0.45, 0.78.",
        "This image features a monument with features 0.67, 0.34, 0.23."
    ]
}

data_t5 = {
    "Image Path": [
        "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg",
        "img6.jpg", "img7.jpg", "img8.jpg", "img9.jpg", "img10.jpg"
    ],
    "Generated Prompt": [
        "Describe an image containing: mountain, river, forest, lake, valley.",
        "Describe an image containing: monument, statue, plaza, temple, castle."
    ]
}

# Extract descriptive terms from prompts
def extract_terms(prompts, keyword="containing:"):
    terms = []
    for prompt in prompts:
        if keyword in prompt:
            terms += prompt.split(keyword)[-1].strip(".").split(", ")
    return Counter(terms)

# Process the data for each method
bayesian_terms = extract_terms(data_bayesian["Prompt"], keyword="most likely a")
clustering_terms = extract_terms(data_clustering["Prompt"], keyword="objects in this cluster, such as")
neural_terms = extract_terms(data_neural["Prompt"], keyword="most likely represents")
template_terms = extract_terms(data_template["Prompt"], keyword="features")
t5_terms = extract_terms(data_t5["Generated Prompt"], keyword="containing:")

# Prepare data for plotting
methods = ["Bayesian", "Clustering", "Neural Network", "Template-Based", "T5 Tokenizer"]
data_counts = [
    len(bayesian_terms),
    len(clustering_terms),
    len(neural_terms),
    len(template_terms),
    len(t5_terms)
]

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(methods, data_counts, color=['blue', 'orange', 'green', 'red', 'purple'])
plt.xlabel("Methods")
plt.ylabel("Unique Descriptive Terms")
plt.title("Comparison of Unique Descriptive Terms by Method")
plt.show()
