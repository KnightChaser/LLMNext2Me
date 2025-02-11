# huggingface_embedding.py
"""
Instead of using the OpenAIEmbeddings class, this example uses the Hugging Face Transformers library to embed the documents.
By doing like this, we can use our own compute resources to embed the documents, which can be useful when we have a large number of documents to embed.
"""

import os
import time
import warnings
from credentials.get_api_credentials import get_api_credentials
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load API credentials and initialize the LangSmith client.
get_api_credentials()

# Adjust the Hugginface model download path with your own path.
os.environ["HF_HOME"] = "/home/knightchaser/cache/HuggingFace"
warnings.filterwarnings("ignore")

sample_texts = [
    "Emotion is the raw, unfiltered force that drives us to defy logic and embrace passion.",
    "Emotion fuels our actions, painting everyday moments with hues of joy, sorrow, and everything in between.",
    "Emotion is the silent current that shapes our decisions—often louder than any rational thought.",
    "Emotion acts as the unseen architect of our lives, building both bridges and barriers with every heartbeat.",
    "In all its messy glory, emotion transforms mere existence into a vibrant, unpredictable journey.",
]

model_name = "intfloat/multilingual-e5-large-instruct"
hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={
        "device": "cpu",
    },  # "cuda" for GPU acceleration and "mps" for multi-processing.
)

start_time = time.time()
embedded_documents = hf_embedding.embed_documents(sample_texts)

print(f"Model: {model_name}")
print(f"Number of documents: {len(sample_texts)}")
print(f"Dimension of the embeddings: {len(embedded_documents[0])}")
print(f"Output of embedded_documents[0]: ...{embedded_documents[0][-3:]}")
print(f"Execution time: {time.time() - start_time:.4f} seconds")
