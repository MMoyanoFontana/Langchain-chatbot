from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document


def build_parent_child_retriever(
    pdf_path,
    embedding_fn,
    persist_dir="chroma_ke_pdf_parent_child_v2",
    collection_name="parent_child_children_v2",
    child_chunk_size=200,
    child_chunk_overlap=60,
    parent_chunk_size=4000,
    parent_chunk_overlap=800,
    parent_separators=["\n## ", "\n# ", "\n\n", "\n", ". "],
    search_kwargs={"k": 4},
):
    # Load and unify pages
    loader = PyMuPDFLoader(pdf_path)
    page_docs = loader.load()
    full_text = "\n".join(d.page_content for d in page_docs)
    docs = [Document(page_content=full_text, metadata={"source": Path(pdf_path).name})]

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=parent_separators,
    )

    # Vectorstore persistente para almacenar los chunks hijos
    # Store en memoria para los padres
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs=search_kwargs,
    )
    retriever.add_documents(docs, ids=None)
    return retriever
