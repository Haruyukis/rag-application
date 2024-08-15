from src.controllers import ssh_everything
from typing import List
from loguru import logger


def eval_everything(
    tasks: List[str], ground_truths: List[int], file_name: str
) -> List[float]:
    """Evaluate every task and make an average"""
    evaluation = []
    for task, ground_truth in zip(tasks, ground_truths):
        logger.info(f"Starting the evaluation for the task: {task}")
        sum = 0
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
                        file.write(f"Output: {len(output_list)}, Ground_Truth: {ground_truth}\n")
            except:
                logger.info("Task failed, trying once again...")
        precision = sum / 5 / ground_truth
        evaluation.append(precision)
    return evaluation


if __name__ == "__main__":
    file_names = ["auth.log", "auth2.log", "auth3.log"]

    tasks = [
        "Please list all distinct users with their username and number of attempts that failed to log in",
        "Please list all users with their username and number of attempts that successfully logged in"
    ]
    ground_truths = [558, 5]
    ground_truths2 = [1536, 11]
    ground_truths3 = [757, 11]

    precision = []
    list_ground_truths = [ground_truths, ground_truths2, ground_truths3]
    file_names.reverse()
    list_ground_truths.reverse()

    for i, file in enumerate(file_names):
        logger.info(f"Starting the evaluation for the file: {file}")
        evaluation = eval_everything(tasks, list_ground_truths[i], file_name=file)
        logger.info(f"Successfully evaluate for the file: {file}")
        precision.append(evaluation)


