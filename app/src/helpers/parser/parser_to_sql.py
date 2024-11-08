def parse_response_to_sql(response: str) -> str:
    """Parse response to SQL."""
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        # TODO: move to removeprefix after Python 3.9+
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:") :]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]
    sql_query = response.strip().strip("```").strip()
    if sql_query[-1] == ";":
        sql_query = sql_query[:-1]
    return sql_query
