import os
import logging

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOGGING_LEVEL = getattr(logging, os.getenv("LOGGING_LEVEL", "DEBUG").upper())
TURN_ON_THIRD_PARTY_LOGGING = os.getenv("TURN_ON_THIRD_PARTY_LOGGING", "false").lower()