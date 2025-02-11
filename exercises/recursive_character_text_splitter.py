# recursive_character_text_splitter.py
"""
RecursiveCharacterTextSplitter is a text splitter that splits the text into chunks based on the number of characters in each chunk.
Almost cases of texts are suitable with RecursiveCharacterTextSplitter.
"""
from typing import List
from pprint import pprint
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("./resource/sample_sales_report.txt") as f:
    file = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # The maximum number of characters in each chunk
    chunk_overlap=50,  # The number of characters to overlap between chunks
    length_function=len,  # The function to use to calculate the length of the text
    is_separator_regex=False,  # If True, the separator is a regex pattern
)

# By using text_splitter, we can `file` text into documents
texts: List[Document] = text_splitter.create_documents([file])
print(texts[0])  # The first chunk of the text
print("=" * 50)
print(texts[1])  # The second chunk of the text

# Or, split the text into chunks
print("=" * 50)
pprint(text_splitter.split_text(file))  # The list of chunks of the text

# page_content='**Cosmic Nexus Aerospace – Q3 Sales Report**'
# ==================================================
# page_content='*Executive Summary*
# Cosmic Nexus Aerospace had a dynamic third quarter marked by solid revenue growth, a few hard lessons, and a clear path forward. Our primary business—manufacturing spacecraft, providing launch services, and managing an array of space-related projects—continues to disrupt a crowded market. We’ve seen encouraging momentum, even if we’ve had to own up to some misfires along the way. No sugar-coating: it’s been a quarter of both significant wins and critical challenges.'
# ==================================================
# ['**Cosmic Nexus Aerospace – Q3 Sales Report**',
#  '*Executive Summary*  \n'
