from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO, filename="splitter.log", filemode="w")

def split_docs(documents):
    # splitting the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1250,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
    splits = text_splitter.split_documents(documents)
    for split in splits:
        logging.info(f"Split ID: {split.metadata.get('source', 'N/A')}, Content: {split.page_content[:100]}...")
    return splits
