from src.ssh_analyse.ssh_svc.ssh_analyzer import SshAnalyzer
import pytest


@pytest.mark.parametrize(
    "user_query, expected_sql_query",
    [
        (
            "Get me all users that failed to log in",
            "select distinct user from failed_logins",
        ),
        (
            "Get me all users that succeed to log in",
            "select distinct user from successful_logins",
        ),
    ],
)
def test_sql_query(user_query: str, expected_sql_query: str):
    """Test SSH Analyzer"""
    ssh_analyzer = SshAnalyzer("./data/log/auth.log")

    ssh_analyzer.run(user_query)
    # Check if the generated sql_query is the right one.
    assert ssh_analyzer.get_sql_query().lower() == expected_sql_query
