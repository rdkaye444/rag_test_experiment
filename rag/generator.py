from schema.document import Document

class Generator:
    def __init__(self):
        self.last_prompt = ""

    def generate(self, query: str, documents: list[Document])-> str:
        llm_boilerplate = "Answer the query based on the following documents:"
        documents_str = "\n".join([doc.data for doc in documents])
        self.last_prompt= f"{llm_boilerplate}\n\n{documents_str}\n\nQuery: {query}"
        return documents[0].data
    
    def get_last_prompt(self)-> str:
        return self.last_prompt
