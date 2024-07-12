from src.ssh_analyse.ssh_svc.ssh_agent import SshAgent
from src.ssh_analyse.ssh_svc.ssh_analyzer import SshAnalyzer


def analyse(user_query: str, path: str):
    analyzer = SshAnalyzer(path)
    return analyzer.run(user_query)


def agent_analyse(user_query: str, path: str):
    return SshAgent(path).run(user_query)
