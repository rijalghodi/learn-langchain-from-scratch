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

# Function to get vector store
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

###### START CREATE VECTOR STORE ######

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, "db", "romeo_and_juliet_db")
file_path = os.path.join(current_dir, "sources", "romeo_and_juliet.txt")

# Create embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Read the text content from the file
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# Split the document into chunks
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create the vector store and persist it automatically
db = init_vector_store(docs, embedding, persist_directory=persist_dir)

###### END CREATE VECTOR STORE ######


###### START CREATE CHAIN AND RETRIEVER ######

retriever = db.as_retriever(
    search_type="similarity", # alternatives: mmr, similarity_score_threshold
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

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()

###### END CHAIN AND RETRIEVER ######

##### START CREATE CHAT CONERSATION #####
def ask_with_session(session_id: str, query: str):
    return rag_chain.invoke(query, 
               config={"configurable": {"session_id": session_id}})
    
conversation = Conversation(ask_with_session=ask_with_session, welcome_text="Welcome to Romeo and Juliet Bot app!")

conversation.chat()

##### END CREATE CHAT #####


