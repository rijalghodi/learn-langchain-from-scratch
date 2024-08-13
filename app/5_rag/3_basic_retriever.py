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
db_dir = os.path.join(current_dir, "db", "sangkuriang_db")
file_path = os.path.join(current_dir, "sources", "sangkuriang.txt")

# Ensure the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# Read the text content from the file
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Create embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Create the vector store and persist it automatically
db = init_vector_store(docs, embedding, db_dir)

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

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

response = rag_chain.invoke("Why Sangkuriang kick the boat?")

print(response.content)
###### END CHAIN AND RETRIEVER ######


