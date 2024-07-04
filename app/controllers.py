from fastapi import FastAPI
from services import generate_response
from ssh_analyse.ssh_svc.main import analyse

app = FastAPI()

@app.get("/api/generate")
def generate(user_query: str):
    response = generate_response(user_query)
    return response.response

@app.get("/api/analyse")
def ssh_analyse(user_query: str, path: str):
    response = analyse(user_query, path)
    return response.response