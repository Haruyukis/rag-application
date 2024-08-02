import os

from sqlalchemy import create_engine, inspect
from src.ssh_analyse.ssh_svc.ssh_analyzer import SshAnalyzer
from src.ssh_analyse.ssh_svc.ssh_database import SshDatabase
from loguru import logger
import runpy


def analyse(user_query: str, path: str):
    analyzer = SshAnalyzer(path)
    return analyzer.run(user_query)


def database(user_query: str, folder_path: str):
    attempts = 0
    ssh_database = SshDatabase(user_query, folder_path)
    while attempts < 3:
        if os.path.exists("text.txt"):
            os.remove("text.txt")
        response = ssh_database.run()
        try:
            with open("draft.py", mode="w") as file:
                file.write(response)
            if os.path.exists("logs.db"):
                os.remove("logs.db")
            runpy.run_path("draft.py")
            break
        except:
            if attempts == 3:
                return "Failed to generate the value..."
            attempts += 1
            logger.info(f"It's the {attempts} attempts...")

    engine = create_engine("sqlite:///logs.db")
    inspector = inspect(engine)
    if len(inspector.get_table_names()) == 0:
        database(user_query, folder_path)
