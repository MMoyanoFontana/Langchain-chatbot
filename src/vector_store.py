from langchain_chroma import Chroma


def initialize_vector_store(
    embeddings_model, collection_name="test", storage_path="chroma_db"
):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_model,
        persist_directory=storage_path,
    )
    return vector_store
