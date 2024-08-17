from src.controllers import ssh_everything
from typing import List
from loguru import logger


def eval_everything(
    tasks: List[str], file_name: str
) -> List[float]:
    """Evaluate every task and make an average"""
    for task in tasks:
        logger.info(f"Starting the evaluation for the task: {task}")
        successfull_attempts = 0
        while successfull_attempts < 5:
            try:
                output = ssh_everything(
                    user_query=task, path="./data/log", file_name=file_name
                )
                output_list = eval(output)
                if isinstance(output_list, list):
                    successfull_attempts += 1
                    sum += len(output_list)
                    logger.info(
                        f"Attempts for the task {task}: {successfull_attempts}, Output: {len(output_list)}, Ground-Truth: {ground_truth}"
                    )
                    with open("evaluation.txt", mode="a") as file:
                        file.write(f"For the task {task} on file {file_name} at the attempts: {successfull_attempts}, the output is:\n")
                        file.write(output_list)
                        file.write("\n")
            except:
                logger.info("Task failed, trying once again...")


if __name__ == "__main__":
    file_names = ["auth.log", "auth2.log", "auth3.log"]

    tasks = [
        "Please list all distinct users with their username and number of attempts that failed to log in",
        "Please list all users with their username and number of attempts that successfully logged in"
    ]
    file_names.reverse()
    list_ground_truths.reverse()

    for i, file in enumerate(file_names):
        logger.info(f"Starting the evaluation for the file: {file}")
        evaluation = eval_everything(tasks, file_name=file)
        logger.info(f"Successfully evaluate for the file: {file}")
        precision.append(evaluation)


