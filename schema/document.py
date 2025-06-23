from pydantic import BaseModel, UUID4

class MetaData(BaseModel):
    title: str
    tags: list[str]
    source: str

class Document(BaseModel):
    id: UUID4
    metadata: MetaData
    data: str




