from pydantic import BaseModel, UUID4, Optional

class Query(BaseModel):
    query: str
    expected_doc_ids: list[UUID4]
    category: str
    expected_semantic_output: str
    adversarial_varients: list[str]
    notes: Optional[str]
