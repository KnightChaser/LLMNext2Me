# flag_embedding_bge_me3.py
"""
This example demonstrates how to use the BGEM3FlagModel from the FlagEmbedding package
to embed a list of short sentences with the BAAI/bge-m3 model (ColBERT-style multi-vector representations).
We then compare each sentence with one another using a max-similarity function.
This way is much easier and more efficient than using the Hugging Face model directly
(take a look at ./huggingface_embedding_bge_m3.py.)
"""

import time
from FlagEmbedding import BGEM3FlagModel

if __name__ == "__main__":
    # Define a list of short, comparison-friendly sentences.
    sample_texts = [
        "Passion fuels creativity.",
        "Emotion inspires art.",
        "Feelings drive action.",
        "Love motivates change.",
        "Desire sparks innovation.",
    ]

    # Initialize the BGEM3FlagModel.
    # Here we use FP16 for faster computation (if supported) and specify the model name.
    # Change device="cuda" if GPU acceleration is available.
    bge_flagmodel = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cpu")

    # Encode the sentences.
    start_time = time.time()
    # The encode method returns a dictionary containing 'colbert_vecs', which are the multi-vector representations.
    bge_encoded = bge_flagmodel.encode(sample_texts, return_colbert_vecs=True)
    elapsed_time = time.time() - start_time
    print("Model: BAAI/bge-m3 (via BGEM3FlagModel with FP16)")
    print(f"Number of sentences: {len(sample_texts)}")
    print(f"Encoding execution time: {elapsed_time:.4f} seconds\n")

    # Compute and display pairwise ColBERT scores.
    # The colbert_score method computes the similarity between two sets of token vectors.
    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            score = bge_flagmodel.colbert_score(
                bge_encoded["colbert_vecs"][i], bge_encoded["colbert_vecs"][j]
            )
            print(
                f"ColBERT score ({score:.4f}) => {sample_texts[i]} <---> {sample_texts[j]}"
            )

# Model: BAAI/bge-m3 (via BGEM3FlagModel with FP16)
# Number of sentences: 5
# Encoding execution time: 0.1071 seconds
#
# ColBERT score (0.7781) => Passion fuels creativity. <---> Emotion inspires art.
# ColBERT score (0.7265) => Passion fuels creativity. <---> Feelings drive action.
# ColBERT score (0.7223) => Passion fuels creativity. <---> Love motivates change.
# ColBERT score (0.7811) => Passion fuels creativity. <---> Desire sparks innovation.
# ColBERT score (0.7702) => Emotion inspires art. <---> Feelings drive action.
# ColBERT score (0.7010) => Emotion inspires art. <---> Love motivates change.
# ColBERT score (0.7103) => Emotion inspires art. <---> Desire sparks innovation.
# ColBERT score (0.7394) => Feelings drive action. <---> Love motivates change.
# ColBERT score (0.7469) => Feelings drive action. <---> Desire sparks innovation.
# ColBERT score (0.7616) => Love motivates change. <---> Desire sparks innovation.
