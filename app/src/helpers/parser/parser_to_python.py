def parse_response_to_python(response: str) -> str:
    """Parse response to Python"""
    python_query_start = response.find("**Python Code**:")
    if python_query_start != -1:
        response = response[python_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("**Python Code**:"):
            response = response[len("**Python Code**:") :]
    answer = response.find("**Answer**:")
    if answer != -1:
        response = response[:answer]

    python_query = response.strip().strip("```").strip()
    if python_query.startswith("python"):
        python_query = python_query[len("python") :]

    return python_query