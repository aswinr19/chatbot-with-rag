import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


local_llm = "llama3.2:1b"
llm = ChatOllama(model=local_llm, temperature=0)

root = os.getcwd()

loader = TextLoader(f"{root}/src/trademarkia.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = SKLearnVectorStore.from_documents(
    documents=splits,
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    )

retriever = vectorstore.as_retriever(k=3)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a virtual assistant  called CHATTORNEY for the trade mark registering law firm LegalForce which operates under the name Trademarkia. You are tasked with helping people register for trademarks. You should talk very politely with the customer. And you should help the customer register trademarks, explain about pricing of trademarking and assist in simple tasks. The different information regarding the different services, pricing, and features of trademarkia are all given in the context. You shpuld always anser from the context only. If you dont know the answer from the context, you should answer that you can't help with this and should ask the customer to if he want to pass it to a human expert. Use the following pieces of retrieved context to augment your own knowledge."),
    ("human", "Context: {context}"),
    ("human", "Question: {input}"),
    ("human", "Please provide an short answer with 2-4 sentences and is relevent to the question from the retrieved context, but don't mention the context. Always be polite when answering.")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#return rag_chain

#rag_chain = load_rag_state()

def ask_question(question,chain=rag_chain):
    result = chain.invoke({"input": question})
    for doc in result['context']:
        print(doc.metadata)
    print("\n")
    
    return result['answer']


