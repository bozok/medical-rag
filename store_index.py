from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_minimal_docs, text_splitter, download_embedding_model
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


extracted_data = load_pdf_files("data/")
filtered_data = filter_minimal_docs(extracted_data)
text_chunks = text_splitter(filtered_data)

embedding = download_embedding_model()

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-rag"

if not pinecone_client.has_index(index_name):
    # Create a new index with the specified name and dimension
    pinecone_client.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
index = pinecone_client.Index(index_name)

doc_search = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embedding
)