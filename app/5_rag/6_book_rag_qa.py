import os
from typing import List
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from app.utils.conversation import Conversation

"""
BOOK RAG QA

This script creates a conversational RAG application, it is an CLI that
run LLM model that can answer your questions based on the text provided.
"""

def init_vector_store(docs: List[Document], embedding: Embeddings, persist_directory: str) -> Chroma:
    """
    Initialize the Chroma vector store, creating it if it doesn't exist.
    """
    if not os.path.exists(persist_directory):
        print("Persistent directory does not exist. Initializing vector store...")
        db = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)
        print("\n--- Finished creating vector store ---")
    else:
        print("Vector store already exists. Loading vector store...")
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    return db

def load_and_split_documents(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Load and split text documents from the given file path into chunks.
    """
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_rag_chain(persist_dir: str, file_path: str) -> ChatOpenAI:
    """
    Create and return the RAG chain model.
    """
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = load_and_split_documents(file_path, chunk_size=1000, chunk_overlap=50)
    db = init_vector_store(docs, embedding, persist_directory=persist_dir)
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    
    model = ChatOpenAI(model="gpt-3.5-turbo")
    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([("human", message)])
    
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()

def ask_with_session(session_id: str, query: str):
    """
    Function to handle chat queries with session management.
    """
    return rag_chain.invoke(query, config={"configurable": {"session_id": session_id}})

def main():
    """
    Main function to set up and start the conversation bot.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(current_dir, "db", "romeo_and_juliet_db")
    file_path = os.path.join(current_dir, "sources", "romeo_and_juliet.txt")
    
    global rag_chain
    rag_chain = create_rag_chain(persist_dir, file_path)
    
    # Chat Loop to interact with the user
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        # Add the user's message to the conversation memory
        memory.chat_memory.add_message(HumanMessage(content=user_input))

        # Invoke the agent with the user input and the current chat history
        response = agent_executor.invoke({"input": user_input})
        print("Bot:", response["output"])

        # Add the agent's response to the conversation memory
        memory.chat_memory.add_message(AIMessage(content=response["output"]))
    
    conversation = Conversation(ask_with_session=ask_with_session, 
                                welcome_text="--- Welcome to Romeo and Juliet Bot QA app ---",
                                human_alias='Question',
                                ai_alias='Answer')
    conversation.chat()

if __name__ == "__main__":
    main()
