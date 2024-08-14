import os
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

"""
BOOK RAG

This script creates a simple RAG application, it is a LLM model
that can answer a question based on the text provided.
"""

def init_vector_store(docs: List[Document], embedding: Embeddings, persist_directory: str) -> Chroma:
    """
    Initialize the Chroma vector store, either by creating it or loading an existing one.
    """
    if not os.path.exists(persist_directory):
        print("Persistent directory does not exist. Initializing vector store...")
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)
        print("\n--- Finished creating vector store ---")
    else:
        print("Vector store already exists. Loading vector store...")
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return db

def load_documents(file_path: str) -> List[Document]:
    """
    Load and split text documents from the given file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    return text_splitter.split_documents(documents)

def create_rag_chain(db_dir: str, file_path: str) -> ChatOpenAI:
    """
    Create and return the RAG chain model.
    """
    docs = load_documents(file_path)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    db = init_vector_store(docs, embedding, db_dir)
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    
    model = ChatOpenAI(model="gpt-3.5-turbo")
    
    prompt = ChatPromptTemplate.from_messages([("human", """
        Answer this question using the provided context only.

        {question}

        Context:
        {context}
    """)])
    
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | model

def main():
    """
    Main function to execute the RAG chain with a sample question.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db", "sangkuriang_db")
    file_path = os.path.join(current_dir, "sources", "sangkuriang.txt")

    rag_chain = create_rag_chain(db_dir, file_path)
    response = rag_chain.invoke("Why did Sangkuriang kick the boat?")
    print(response.content)

if __name__ == "__main__":
    main()
