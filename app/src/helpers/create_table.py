from typing import List
from sqlalchemy import MetaData, Table, Column
import re


def create_table_from_data(
    path: str,
    table_names: List[str],
    metadata_obj: MetaData,
    columns: List[List[Column]],
    engine,
):
    pattern_failed = r"^(\w{3} \d{2} \d{2}:\d{2}:\d{2}) \S+ sshd\[\d+\]: Connection closed by (?:authenticating|invalid) user (\S+) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) port (\d+) \[preauth\]$"
    pattern_succeed = r"^(\w{3}\s+\d{1,2} \d{2}:\d{2}:\d{2}) \S+ sshd\[\d+\]: Accepted publickey for (\S+) from (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) port (\d+)"
    # Creating all table with the defined columns
    tables = []
    for i, table_name in enumerate(table_names):
        tables.append(Table(table_name, metadata_obj, *(columns[i])))

    # Create the table in the db.
    metadata_obj.create_all(engine)

    with open(path, "r") as file:
        with engine.connect() as conn:
            for line in file:
                match_failed = re.search(pattern_failed, line)
                match_succeed = re.search(pattern_succeed, line)

                if match_failed:
                    username = match_failed.group(2)
                    attempt_time = match_failed.group(1)
                    insert_stmt = (
                        tables[1]
                        .insert()
                        .values(
                            user=username,
                            attempt_time=attempt_time,
                            log_message=line,
                        )
                    )
                    conn.execute(insert_stmt)
                elif match_succeed:
                    username = match_succeed.group(2)
                    login_time = match_succeed.group(1)
                    insert_stmt = (
                        tables[0]
                        .insert()
                        .values(
                            user=username,
                            login_time=login_time,
                            log_message=line,
                        )
                    )
                    conn.execute(insert_stmt)
            conn.commit()
