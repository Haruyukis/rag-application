from src.ssh_analyse.ssh_svc.main import database, analyse
import pytest


@pytest.mark.parametrize("user_query", ["Please list all users that failed to log in"])
def test_everything_query(user_query: str):
    response = database(user_query, "./data/log")
    if response is None:
        return "Failed to proceed to the task..."
    return analyse(user_query, "./data/log/auth.log")
