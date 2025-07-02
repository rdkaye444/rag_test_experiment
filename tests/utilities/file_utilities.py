import json
import logging
from pathlib import Path

from pydantic import BaseModel

TEST_DATA_DIR = Path("tests/data")
logger = logging.getLogger(__name__)

def _load_jsonl(file_path: str):
    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    yield json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.error(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"Warning: Test data file {file_path} not found. Tests may fail.")
        return


def load_test_data(file_name:str, schema:BaseModel):
    """
    Load test data from a JSONL file and return a list of instances of the given schema.
    The file is expected to be in the tests/data directory.
    """
    return [schema(**data) for data in _load_jsonl(f"{TEST_DATA_DIR}/{file_name}")]