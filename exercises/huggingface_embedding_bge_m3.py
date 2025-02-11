# huggingface_embedding_bge_m3.py
"""
This example demonstrates how to use the Hugging Face model "BAAI/bge-m3" for embedding texts.
This model produces multi-vector representations (ColBERT-style), which means that each text is
represented as a collection of token embeddings. We then compute a sentence-level similarity using
a max-sim approach.
"""

import os
import time
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from credentials.get_api_credentials import get_api_credentials
from langsmith import Client, traceable

# Load API credentials and initialize the LangSmith client.
get_api_credentials()
langsmith_client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

# Set up Hugging Face cache directory (modify the path as needed) and suppress warnings.
os.environ["HF_HOME"] = "/home/knightchaser/cache/HuggingFace"
warnings.filterwarnings("ignore")

# Define model and device settings.
model_name = "BAAI/bge-m3"
device = "cpu"  # Change to "cuda" for GPU acceleration if available.

# Load the tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


def embed_text_multi_vector(text: str) -> np.ndarray:
    """
    Embeds a single text into multiple vectors using the BAAI/bge-m3 model.

    Args:
        text (str): The input text.

    Returns:
        np.ndarray: A 2D array of shape (num_tokens, embedding_dim) containing the token embeddings.
    """
    # Tokenize the input text and move the inputs to the specified device(e.g. CPU or GPU).
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model outputs.
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Use the last hidden state as token embeddings (shape: [1, seq_len, hidden_size]).
    token_embeddings = outputs.last_hidden_state.squeeze(
        0
    )  # Now shape: (seq_len, hidden_size)

    # Filter out padding tokens using the attention mask.
    mask = inputs["attention_mask"].squeeze(0).bool()  # Shape: (seq_len,)
    token_embeddings = token_embeddings[mask]

    # Normalize the token embeddings (L2 normalization), which is useful for cosine similarity.
    token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)

    return token_embeddings.cpu().numpy()


@traceable(client=langsmith_client)
def embed_documents_multi_vector(texts: list) -> list:
    """
    Embeds a list of texts into multi-vector representations.

    Args:
        texts (list): A list of strings, where each string is a document or sentence.

    Returns:
        list: A list where each element is a 2D numpy array of token embeddings for a text.
    """
    return [embed_text_multi_vector(text) for text in texts]


def maxsim_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    """
    Computes a similarity score between two sets of token embeddings using the max-sim approach.

    For each vector in one set, it finds the maximum cosine similarity with the vectors in the other set,
    and then averages these maximum values.

    Args:
        embeddings1 (np.ndarray): Array of shape (n1, d) for the first text.
        embeddings2 (np.ndarray): Array of shape (n2, d) for the second text.

    Returns:
        float: The computed similarity score.
    """
    # Compute cosine similarity matrix between all pairs of token embeddings.
    # similarities = query * document^T
    # (Since embeddings are already normalized, a dot product yields cosine similarity.)
    sim_matrix = np.dot(embeddings1, embeddings2.T)  # Shape: (n1, n2)

    # For each token in embeddings1, get the max similarity with any token in embeddings2.
    max_sim_1 = sim_matrix.max(axis=1)
    # For each token in embeddings2, get the max similarity with any token in embeddings1.
    max_sim_2 = sim_matrix.max(axis=0)

    # Average the maximum similarities.
    return (max_sim_1.mean() + max_sim_2.mean()) / 2


if __name__ == "__main__":
    # Define some sample texts.
    sample_texts = [
        "Emotion is the raw, unfiltered force that drives us to defy logic and embrace passion.",
        "Emotion fuels our actions, painting everyday moments with hues of joy, sorrow, and everything in between.",
        "Emotion is the silent current that shapes our decisions—often louder than any rational thought.",
        "Emotion acts as the unseen architect of our lives, building both bridges and barriers with every heartbeat.",
        "In all its messy glory, emotion transforms mere existence into a vibrant, unpredictable journey.",
    ]

    # Embed the sample texts.
    start_time = time.time()
    embedded_documents = embed_documents_multi_vector(sample_texts)
    elapsed_time = time.time() - start_time

    # Display embedding information.
    print(f"Model: {model_name}")
    print(f"Number of documents: {len(sample_texts)}")
    print(f"Document 1 embedding shape: {embedded_documents[0].shape}")
    print(f"Last 3 vectors of Document 1:\n{embedded_documents[0][-3:]}")
    print(f"Embedding execution time: {elapsed_time:.4f} seconds\n")

    # Demonstrate similarity computation for sample_texts each other
    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            sim_score = maxsim_similarity(embedded_documents[i], embedded_documents[j])
            print(
                f"MaxSim similarity({sim_score:.4f}) => {sample_texts[i]} vs. {sample_texts[j]}"
            )
