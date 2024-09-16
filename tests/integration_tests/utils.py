import os


def get_abs_path(code_path, relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(code_path), relative_path))
