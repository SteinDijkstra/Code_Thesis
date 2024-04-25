from pathlib import Path
import sys


def get_absolute_path() -> Path:
    return Path("enter path here")


def add_project_to_path():
    base = get_absolute_path()
    sys.path.insert(0, base)
