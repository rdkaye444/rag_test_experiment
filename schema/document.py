from pydantic import BaseModel, UUID4

class MetaData(BaseModel):
    title: str
    source_species: str
    data_source: str

class Document(BaseModel):
    id: str
    metadata: MetaData
    data: str
    rank: float = 0.0




