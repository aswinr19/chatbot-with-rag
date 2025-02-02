from langchain_community.document_loaders import  TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len)
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

vectorstore = Chroma(persist_directory="./chroma_db",embedding_function=embeddings)

def load_and_split_document(file_path: str) -> list[Document]:
    loader = TextLoader(file_path)
    documents = loader.load()

    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        
        for split in splits:
            split.metadata['file_id'] = file_id

        vectorstore.add_documents(splits)

        return True

    except Exception as e:
        print(f"error indexing documents {e}")
        return False




