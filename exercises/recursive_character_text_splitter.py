# recursive_character_text_splitter.py
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
