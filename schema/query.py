from pydantic import BaseModel, UUID4
from typing import Optional

class Query(BaseModel):
    query: str
    expected_doc_ids: list[UUID4]
    category: str
    expected_semantic_output: str
    adversarial_variants: list[str]
    notes: Optional[str]
