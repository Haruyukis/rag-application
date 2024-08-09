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
            output = ssh_everything(
                user_query=task, path="./data/log", file_name=file_name
            )
            try:
                output_list = eval(output)
                successfull_attempts += 1
                sum += len(output_list)
                logger.info(
                    f"Attempts for the task {task}: {successfull_attempts}, Output: {len(output_list)}, Ground-Truth: {ground_truth}"
                )
            except:
                logger.info("Task failed, trying once again...")
        precision = sum / 5 / ground_truth
        evaluation.append(precision)
    return evaluation


if __name__ == "__main__":
    file_names = ["auth.log", "auth2.log"]
    tasks = [
        "Please list all users with their username that failed to log in",
        "Please list all users with their username that successfully logged in",
    ]
    ground_truths = [2831, 190]
    ground_truths2 = [7319, 221]

    precision = []
    list_ground_truths = [ground_truths, ground_truths2]
    for i, file in enumerate(file_names):
        logger.info(f"Starting the evaluation for the file: {file}")
        evaluation = eval_everything(tasks, list_ground_truths[i], file_name=file)
        logger.info(f"Successfully evaluate for the file: {file}")
        precision.append(evaluation)
    with open("evaluation.txt", mode="w") as f:
        f.write(str(precision))
