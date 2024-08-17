import runpy


def run_python(python_code: str, path: str):
    """Run a python code from string"""
    path = path + ".py"
    with open(path, mode="w") as file:
        file.write(python_code)
    runpy.run_path(path)
