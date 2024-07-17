from fastapi import FastAPI
from src.services import generate_response, test
from src.ssh_analyse.ssh_svc.main import analyse, agent_analyse

app = FastAPI()


@app.get("/api/generate")
def generate(user_query: str):
    response = generate_response(user_query)
    return response.response


@app.get("/api/analyse")
def ssh_analyse(user_query: str, path: str):
    """Analyzing ssh log data"""
    return analyse(user_query, path)


@app.get("/api/agent")
def ssh_agent(user_query: str, path: str):
    """Analyzing ssh log data with Agent"""
    return agent_analyse(user_query, path)


@app.get("/api/sentence_splitter")
def sentence_splitter(user_query: str, path: str):
    """Analyzing ssh log data with Agent"""
    return test(user_query)
