from fastapi import FastAPI
from src.services import generate_response
from src.ssh_analyse.ssh_svc.main import analyse, database


app = FastAPI()


@app.get("/api/generate")
def generate(user_query: str):
    response = generate_response(user_query)
    return response.response


@app.get("/api/analyse")
def ssh_analyse(user_query: str, path: str):
    """Analyzing ssh log data"""
    return analyse(user_query, path)


@app.get("/api/database")
def ssh_database(user_query: str, path: str):
    """Query Engine with sentence splitter for auth.log"""
    return database(user_query, path)
