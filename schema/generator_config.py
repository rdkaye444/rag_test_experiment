from typing import Literal, Optional

from pydantic import UUID4, BaseModel


class GeneratorConfig(BaseModel):
    mode: Literal["loose", "strict"]
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
