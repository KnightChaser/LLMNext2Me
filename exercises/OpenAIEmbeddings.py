# OpenAIEmbeddings.py
"""
With using OpenAIEmbeddings, you can get the embeddings of the text.
Embeddings is a technique used to represent the text in a numerical format.
Thus, the computer can understand the text and perform various operations on it.
(Find the similarity between the texts, find the distance between the texts, etc.)
"""

from credentials.get_api_credentials import get_api_credentials
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

get_api_credentials()

# A. Embed the text using OpenAIEmbeddings.
sample_text = "Embedding is imperative for understanding the text for computers."
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query_result = embeddings.embed_query(sample_text)
print(f"Original text: {sample_text}")
print(f"Embeddings: ..., {','.join(map(str, query_result[-3:]))}")
print("=" * 50)

# B. Emed the text document
documents = embeddings.embed_documents(
    [
        # Sampel text documents
        "All I need is a good book to read.",
        "When I was a child, I used to play with my friends.",
        "Now, I am a grown-up and I have to work.",
    ]
)
print(f"Length of the document vectors: {len(documents)}")
print(f"Document 1: ..., {','.join(map(str, documents[0][-3:]))}")
print(f"Document 2: ..., {','.join(map(str, documents[1][-3:]))}")
print(f"Document 3: ..., {','.join(map(str, documents[2][-3:]))}")
print("=" * 50)

# C. Adjusting dimension
print(f"Dimension of original text embeddings: {len(query_result)}")
embeddings_1024 = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
print(
    f"Dimension of original text embeddings(adjusted): {len(embeddings_1024.embed_query(sample_text))}"
)
print("=" * 50)

# D. Calculating semantic similarities
sentences = [
    "Daniel works as a software engineer in New York.",
    "Daniel is a software engineer in New York.",
    "Daniel is a software engineer in New Jersey.",
    "Does Daniel work as a software engineer in New York?",
    "Daniel is a software engineer.",
]


def similarity(source, target) -> float:
    return cosine_similarity([source], [target])[0][0]


embedded_sentences = embeddings_1024.embed_documents(sentences)
for source_index, source_embed in enumerate(embedded_sentences):
    for target_index, target_embed in enumerate(embedded_sentences):
        if source_index < target_index:
            print(
                f"similarity({similarity(source_embed, target_embed):.4f}): {sentences[source_index]} | {sentences[target_index]}"
            )

# Original text: Embedding is imperative for understanding the text for computers.
# Embeddings: ..., -0.0036423944402486086,0.010304100811481476,0.026634426787495613
# ==================================================
# Length of the document vectors: 3
# Document 1: ..., -4.518126388575183e-06,-0.02247878909111023,-0.010205887258052826
# Document 2: ..., -0.02159697562456131,-0.008885071612894535,0.008158857934176922
# Document 3: ..., 0.012924771755933762,-0.0020148116163909435,-0.02663356252014637
# ==================================================
# Dimension of original text embeddings: 1536
# Dimension of original text embeddings(adjusted): 1024
# ==================================================
# similarity(0.9514): Daniel works as a software engineer in New York. | Daniel is a software engineer in New York.
# similarity(0.8755): Daniel works as a software engineer in New York. | Daniel is a software engineer in New Jersey.
# similarity(0.8882): Daniel works as a software engineer in New York. | Does Daniel work as a software engineer in New York?
# similarity(0.8336): Daniel works as a software engineer in New York. | Daniel is a software engineer.
# similarity(0.9208): Daniel is a software engineer in New York. | Daniel is a software engineer in New Jersey.
# similarity(0.8857): Daniel is a software engineer in New York. | Does Daniel work as a software engineer in New York?
# similarity(0.8690): Daniel is a software engineer in New York. | Daniel is a software engineer.
# similarity(0.7990): Daniel is a software engineer in New Jersey. | Does Daniel work as a software engineer in New York?
# similarity(0.8442): Daniel is a software engineer in New Jersey. | Daniel is a software engineer.
# similarity(0.7645): Does Daniel work as a software engineer in New York? | Daniel is a software engineer.
