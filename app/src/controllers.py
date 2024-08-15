from fastapi import FastAPI
from src.services import generate_response
from src.ssh_analyse.ssh_svc.main import analyse, database

app = FastAPI()


@app.get("/api/generate")
def generate(user_query: str):
    response = generate_response(user_query)
    return response.response


@app.get("/api/analyse")
def ssh_analyse(user_query: str):
    """Analyzing ssh log data"""
    return analyse(user_query)


@app.get("/api/database")
def ssh_database(user_query: str, path: str, file_name: str):
    """Query Engine with sentence splitter for auth.log"""
    return database(user_query, path, file_name)


@app.get("/api/everything")
def ssh_everything(user_query: str, path: str, file_name):
    """Create the database and analyse"""
    without_distinct = user_query.replace("distinct", "").strip()
    response = database(without_distinct, path, file_name)
    if response is not None:
        return response
    return analyse(user_query)
