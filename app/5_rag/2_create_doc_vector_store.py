import os
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# DEFINE FUNCTION TO GET AND INIT VECTOR STORE 
def init_vector_store(docs: List[Document], embedding: Embeddings, persist_directory: str = None) -> None:
    # Check if the Chroma vector store already exists    
    if not os.path.exists(persist_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        # Create the vector store and persist it automatically
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(
            docs, embedding, persist_directory=persist_directory)
        print("\n--- Finished creating vector store ---")
        return db

    else:
        print("Vector store already exists. No need to initialize.")
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        return db

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "sources", "romeo_and_juliet.txt")
persist_directory = os.path.join(current_dir, "book-db")

# Ensure the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path)
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings
print("\n--- Creating embeddings ---")
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small"
)  # Update to a valid embedding model if needed
print("\n--- Finished creating embeddings ---")

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")

# Create vector store
vector_store = init_vector_store(docs, embedding, persist_directory)