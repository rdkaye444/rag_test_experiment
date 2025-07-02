import os
import logging

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOGGING_LEVEL = getattr(logging, os.getenv("LOGGING_LEVEL", "DEBUG").upper())
THIRD_PARTY_LOGGING_LEVEL = getattr(logging, os.getenv("THIRD_PARTY_LOGGING_LEVEL", "WARNING").upper())