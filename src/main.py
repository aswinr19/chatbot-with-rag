import os
import requests
from rag import ask_question
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, HTTPException, Request, Response , Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OLLAMA_SERVER_URL = "http://localhost:11434"

class Query(BaseModel):
    prompt: str
    model: str

class RaggedModel:
    def __init__(self, model):
        #local_llm = "llama3.2:1b"
        self.local_llm = model
        self.llm = ChatOllama(model=self.local_llm, temperature=0)

    def load_model(self):
        self.root = os.getcwd()
        self.loader = TextLoader(f"{self.root}/src/trademarkia.txt")
        self.docs = self.loader.load()
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = self.text_splitter.split_documents(self.docs)
        
        self.vectorstore = SKLearnVectorStore.from_documents(
            documents=self.splits,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
            )
        self.retriever = self.vectorstore.as_retriever(k=3)
       
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a virtual assistant  called CHATTORNEY for the trade mark registering law firm LegalForce which operates under the name Trademarkia. You are tasked with helping people register for trademarks. You should talk very politely with the customer. And you should help the customer register trademarks, explain about pricing of trademarking and assist in simple tasks. The different information regarding the different services, pricing, and features of trademarkia are all given in the context. You shpuld always anser from the context only. If you dont know the answer from the context, you should answer that you can't help with this and should ask the customer to if he want to pass it to a human expert. Use the following pieces of retrieved context to augment your own knowledge."),
            ("human", "Context: {context}"),
            ("human", "Question: {input}"),
            ("human", "Please provide an short answer with 2-4 sentences and is relevent to the question from the retrieved context, but don't mention the context. Always be polite when answering.")
        ])
        
        self.question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
        #return rag_chain
        #rag_chain = load_rag_state()
        
    def ask_question(self,question):
        result = self.rag_chain.invoke({"input": question})
        return result['answer']
  
    def close(self):
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = RaggedModel(model="llama3.2:1b")
    await model.load_model()
    yield { "model": model }
    app.state.model = model
    await model.close()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def health_check(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.post("/generate")
async def generate_response(query: Query):
    try:
        print(query.prompt)
        model: RaggedModel = app.state.model;
        print(model)
        response = model.ask_question(question=query.prompt)
        print(f"{query.prompt}: {response}")

        return { "generated_text": response }
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error while trying to communicate to ollama {str(e)}")

@app.get("/models")
async def list_models():

    try:
        ollama_api_url = f"{OLLAMA_SERVER_URL}/api/tags"

        response = requests.get(ollama_api_url)

        response.raise_for_status()
        
        return { "models": response.json()["models"] }
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models {str(e)}")

