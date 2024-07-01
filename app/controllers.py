from fastapi import FastAPI
from services import generate_response

app = FastAPI()

@app.get("/api/generate")
def generate(user_query: str):
    response = generate_response(user_query)
    return response.response
