# semantic_chunker.py
"""
SemanticChunker is a text splitter that splits the text into chunks based on the semantic meaning of the text.
It does not split the text based on the number of characters in each chunk.
It measures the semantic meaning of the text and splits the text into chunks based on the semantic meaning(such as if the text has more semantic distance than the threshold, the text is split.).
"""

from credentials.get_api_credentials import get_api_credentials
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

with open("./resource/sample_sales_report.txt") as f:
    file = f.read()

get_api_credentials()

# Initialize SemanticChunker via OpenAIEmbeddings
text_splitter = SemanticChunker(OpenAIEmbeddings())

# Split the text (via text chunk)
chunks = text_splitter.split_text(file)
print(chunks[0])
print(f"{type(chunks)} => {type(chunks[0])}")
print("=" * 50)

# Split the text (via document chunk)
documents = text_splitter.create_documents([file])
print(documents[0].page_content)
print(f"{type(documents)} => {type(documents[0])} => {type(documents[0].page_content)}")
print("=" * 50)

# Use SemanticChunker with standard deviation statistics.
# If the semantic difference(distance) exceeds the given standard deviation, the text is split.
text_splitter = SemanticChunker(
    # Use OpenAIEmbeddings to initialize the SemanticChunker
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5,
)

documents = text_splitter.create_documents([file])
for index, document in enumerate(documents):
    print(f"Document {index + 1}\n")
    print(document.page_content)
    print("-" * 50)

# **Cosmic Nexus Aerospace – Q3 Sales Report**
#
# *Executive Summary*
# Cosmic Nexus Aerospace had a dynamic third quarter marked by solid revenue growth, a few hard lessons, and a clear path forward. Our primary business—manufacturing spacecraft, providing launch services, and managing an array of space-related projects—continues to disrupt a crowded market.
# <class 'list'> => <class 'str'>
# ==================================================
# **Cosmic Nexus Aerospace – Q3 Sales Report**
#
# *Executive Summary*
# Cosmic Nexus Aerospace had a dynamic third quarter marked by solid revenue growth, a few hard lessons, and a clear path forward. Our primary business—manufacturing spacecraft, providing launch services, and managing an array of space-related projects—continues to disrupt a crowded market.
# <class 'list'> => <class 'langchain_core.documents.base.Document'> => <class 'str'>
# ==================================================
# Document 1
#
# **Cosmic Nexus Aerospace – Q3 Sales Report**
#
# *Executive Summary*
# Cosmic Nexus Aerospace had a dynamic third quarter marked by solid revenue growth, a few hard lessons, and a clear path forward. Our primary business—manufacturing spacecraft, providing launch services, and managing an array of space-related projects—continues to disrupt a crowded market.
# --------------------------------------------------
# ...
