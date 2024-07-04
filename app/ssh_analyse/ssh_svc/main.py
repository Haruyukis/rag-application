from ssh_analyse.ssh_svc.SshAnalyzer import SshAnalyzer

def analyse(user_query: str, path: str):
    analyzer = SshAnalyzer(path)

    return analyzer.run(user_query)