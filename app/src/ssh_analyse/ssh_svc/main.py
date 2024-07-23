from src.ssh_analyse.ssh_svc.ssh_analyzer import SshAnalyzer
from src.ssh_analyse.ssh_svc.ssh_database import SshDatabase


def analyse(user_query: str, path: str):
    analyzer = SshAnalyzer(path)
    return analyzer.run(user_query)


def database(user_query: str, folder_path: str):
    database = SshDatabase(user_query, folder_path)
    return database.run()
