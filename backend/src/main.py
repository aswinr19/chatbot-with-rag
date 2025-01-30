from fastapi import FastAPI, HTTPException, Request, Response 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
from rag import ask_question

app = FastAPI()

OLLAMA_SERVER_URL = "http://localhost:11434"


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class Query(BaseModel):
    prompt: str
    model: str

@app.get("/", response_class=HTMLResponse)
async def health_check(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.post("/generate")
async def generate_response(query: Query):
    try:
        response = ask_question(question=query.prompt)

        print(f"{query.prompt}: {response}")

        return { "generated_text": response }
        #ollama_api_url = f"{OLLAMA_SERVER_URL}/api/generate"
        #json_payload = {"model": query.model, "prompt": query.prompt, "stream": False }

        #response = requests.post(
        #    ollama_api_url,
        #    json=json_payload
        #)

        #print(response)

        #response.raise_for_status()

        #return { "generated_text": response.json()["response"] }

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


