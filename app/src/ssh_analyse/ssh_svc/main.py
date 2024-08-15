import os

from sqlalchemy import create_engine, inspect
from src.helpers.execute_text2python import run_python
from src.helpers.parser.llm_parser_to_python import parse_using_llm
from src.ssh_analyse.ssh_svc.ssh_analyzer import SshAnalyzer
from src.ssh_analyse.ssh_svc.ssh_database import SshDatabase
from loguru import logger

def analyse(user_query: str):
    analyzer = SshAnalyzer()
    return analyzer.run(user_query)


def database(user_query: str, folder_path: str, file: str):
    """Generate & Execute the python code"""
    drop_empty_table_prompt = """\
\n\n
from sqlalchemy import inspect
# Function to drop empty tables
def drop_empty_tables(engine, Base):
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():

        table = Base.metadata.tables[table_name]
        if session.query(table).first() is None:
            print(f"Dropping empty table: {table_name}")
            table.drop(engine)

# Drop empty tables
drop_empty_tables(engine, Base)
"""
    ssh_database = SshDatabase(user_query, folder_path, file)
    attempts = 0
    while attempts < 3:
        response = ssh_database.run()
        try:
            if os.path.exists("logs.db"):
                os.remove("logs.db")

            run_python(response + drop_empty_table_prompt, path="draft")
            break
        #except:
        #    corrected_response = parse_using_llm(response)
        #    try:
        #        if os.path.exists("logs.db"):
        #            os.remove("logs.db")
        #        run_python(corrected_response + drop_empty_table_prompt, path="draft")
        #        break
        except:
            attempts += 1
            logger.info(f"The LLM failed to generate the database. {attempts} times")
            if attempts == 3:
                logger.info("Maximum attempts reached")
                return "The LLM failed to generated the database..."
    engine = create_engine("sqlite:///logs.db")
    inspector = inspect(engine)
    if len(inspector.get_table_names()) == 0:
        database(user_query, folder_path, file)
