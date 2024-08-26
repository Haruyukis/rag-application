from src.controllers import ssh_everything
from typing import List, Tuple
from loguru import logger


def eval_everything(
    tasks: List[Tuple[str]], file_name: str
) -> List[float]:
    """Evaluate every task and make an average"""
    for task in tasks:
        logger.info(f"Starting the evaluation for the task: {task[1]}")
        successfull_attempts = 0
        while successfull_attempts < 5:
            try:
                output = ssh_everything(
                    user_query_database=task[0], user_query_analyzer=task[1], path="./data/log", file_name=file_name
                )
                logger.info("Output received, trying to evaluate")
                output_list = eval(output)
                logger.info("Evaluation done.")
                if isinstance(output_list, list):
                    successfull_attempts += 1
                    logger.info(
                        f"Attempts for the task {task[1]}: {successfull_attempts} for the file {file_name}"
                    )
                    logger.info("Starting to write inside the evaluation")
                    with open("evaluation.txt", mode="a") as file:
                        file.write(f"For the task {task[0]} on file {file_name} at the attempts: {successfull_attempts}, the output is:\n")
                        file.write(str(output_list))
                        file.write("\n")
                        file.flush()
                    logger.info("Done writing...")
            except:
                logger.info("Task failed, trying once again...")
                successfull_attempts += 1
                with open("evaluation.txt", mode="a") as file:
                    file.write(f"For the task {task[0]} on file {file_name} at the attempts: {successfull_attempts}, the output is:")
                    file.write("[Failed to generate the output]")
                    file.write("\n")
                    file.flush()


if __name__ == "__main__":
    file_names = ["auth.log", "auth2.log", "auth3.log"]

    tasks = [("Please store all usernames that failed to log in", "Please list all distinct usernames that failed to log in with their number of attempts"), 
        ("Please store all usernames that successfully logged in", "Please list all distinct usernames that successfully logged in"),
        ("Please store all usernames that have their publickey accepted", "Please list all distinct usernames that have their publickey accepted"),
    ]


    for i, file in enumerate(file_names):
        logger.info(f"Starting the evaluation for the file: {file}")
        eval_everything(tasks, file_name=file)
        logger.info(f"Successfully evaluate for the file: {file}")

    # For the whitelist
    tasks = [("Here is a whitelist of authorized users: ['RAIN', 'SHUKA']. Please store all usernames that successfully logged in without being authorized", "Here is a whitelist of authorized users: ['RAIN', 'SHUKA']. Please list all distinct usernames that successfully logged in without being authorized"),
        ("Here is a whitelist of authorized users: ['YAMAMOTO', 'MORI', 'SAEKI']. Please store all usernames that successfully logged in without being authorized", "Here is a whitelist of authorized users: ['YAMAMOTO', 'MORI', 'SAEKI']. Please list all distinct usernames that successfully logged in without being authorized"),
        ("Here is a whitelist of authorized users: ['SHIMURA', 'TAKASUGI', 'HASEGAWA', 'KAMUI', 'MURATA']. Please store all usernames that successfully logged in without being authorized", "Here is a whitelist of authorized users: ['SHIMURA', 'TAKASUGI', 'HASEGAWA', 'KAMUI', 'MURATA']. Please list all distinct usernames that successfully logged in without being authorized"),
    ]

    for file, task in zip(file_names, tasks):
        logger.info(f"Starting the evaluation for the file {file} and the task {task}")
        eval_everything([task], file_name=file)
        logger.info(f"Starting the evaluation for the file {file} and the task {task}")
    eval_everything([("Please store all usernames that have their password accepted", "Please list all distinct usernames that have their password accepted")], file_name="auth2.log")
        

