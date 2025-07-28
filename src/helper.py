from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import List
from langchain.schema import Document

from langchain.embeddings import HuggingFaceEmbeddings


# Extract text from PDF files
def load_pdf_files(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents



def filter_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: list[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(
            page_content=doc.page_content,
            metadata={"source": src}
        ))
    return minimal_docs


def text_splitter(minimal_docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(minimal_docs)
    return chunks


def download_embedding_model() -> HuggingFaceEmbeddings:
    """
    Download and return HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings_model