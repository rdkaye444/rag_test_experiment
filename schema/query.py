from typing import Optional

from pydantic import UUID4, BaseModel


class Query(BaseModel):
    query: str
    expected_doc_ids: list[UUID4]
    category: str
    expected_semantic_output: str
    adversarial_variants: list[str]
    notes: Optional[str]
