## Chattorney - Virtual Assistant for Trademarking

Chattorney is an intelligent virtual assistant designed to help you with common trademarking queries. Powered by the Llama 3.2 1b model, Chattorney is here to simplify the trademarking process and provide answers to your trademark-related questions quickly and efficiently.

### Features

- Answer common trademarking queries
- Assist with trademark registration and protection
- Powered by Llama 3.2 1b model for accurate responses
- Supports interactive chat via FastAPI

### Requirements

To run this project, make sure you have the following installed:

1) Python 3.10 or greater
2) Ollama (for using the Llama model)
3) langchain
4) langchain_chroma
5) langchain_nomic
6) fastapi[standard]
7) nomic[local]
8) langchain_ollama
9) tiktoken

Installation and Setup

Step 1: Install Python 3.10 or greater

Make sure you have Python 3.10 or greater installed on your system. You can download it from the official Python website.

Step 2: Install Required Libraries

```
Clone the repository and navigate to the project folder: 
git clone git@github.com:aswinr19/chatbot-with-rag.git
cd chatbot-with-rag
```

Create a virtual environment (optional but recommended):

```
python3 -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows

Install the required libraries:
pip install -r requirements.txt
```

Step 3: Install Ollama

To use the Llama 3.2 1b model, you’ll need to install Ollama, which provides access to the model. Follow the installation instructions on the Ollama website for your operating system.

Step 4: Run the Application

Once all dependencies are installed and Ollama is set up, you can start the virtual assistant using FastAPI:
```
fastapi dev main.py
```
This will start the server locally. You can now access the assistant at http://127.0.0.1:8000.

Step 5: Interact with Chattorney

Open a web browser and go to http://127.0.0.1:8000. You will be able to chat with Chattorney and get assistance with your trademark queries.

