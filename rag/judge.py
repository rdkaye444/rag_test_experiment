"""
This module contains the Judge class, which is used to judge the quality of the generated response.
"""

from rag.llm import OpenAI_LLM
from schema.document import Document
from enum import Enum

MODE_JUDGE = "judge"
MODE_EXPLAIN = "explain"


prompts = {
    MODE_JUDGE: """You are a helpful and objective query response evaluator.  
        You will find a set of context documents listed below.  You will also find
        a generated answer.  Does the generated answer contain any factual claims that are
        not explicitly stated in the context documents?  If so, return "False".  
        If not, return "True".
        
        Context documents:
        {context_section}

        Generated answer:
        * {response}""",
    MODE_EXPLAIN: """You are a helpful and objective query response evaluator.  
        You will find a set of context documents listed below.  You will also find
        a generated answer. Explain whether the generated answer is supported by the 
        context or not, and if not, identify the unsupported (hallucinated) parts.

        Context documents:
        {context_section}

        Generated answer:
        * {response}"""
}

class JudgeResult(Enum):
    """
    Enum representing the possible outcomes of a judge evaluation.
    
    Attributes:
        TRUE: The response is supported by the context documents
        FALSE: The response contains unsupported factual claims
        MAYBE: The response is ambiguous or unclear
    """
    TRUE = "true"
    FALSE = "false"
    MAYBE = "maybe"

    def is_definitive(self) -> bool:
        """
        Check if the judge result is definitive (either TRUE or FALSE).
        
        Returns:
            bool: True if the result is definitive, False if it's MAYBE
        """
        return self in (JudgeResult.TRUE, JudgeResult.FALSE)
    
class Judge:
    """
    A class for evaluating the quality and accuracy of generated responses 
    against provided context documents using an LLM-based evaluation approach.
    
    The Judge can operate in two modes:
    - JUDGE: Returns a binary evaluation (TRUE/FALSE/MAYBE)
    - EXPLAIN: Returns a detailed explanation of the evaluation
    
    Attributes:
        llm: The language model used for evaluation
        last_prompt: The last prompt sent to the LLM
        last_result: The last result received from the LLM
    """
    
    def __init__(self):
        """
        Initialize the Judge with an OpenAI LLM instance.
        
        Sets up the LLM client and initializes tracking variables for
        the last prompt and result.
        """
        self.llm = OpenAI_LLM()
        self.last_prompt = ""
        self.last_result = ""

    def _judge(self, response: str, context_documents: list[Document], mode: str = MODE_JUDGE) -> str:
        """
        Internal method to perform the actual LLM-based evaluation.
        
        Args:
            response: The generated response to evaluate
            context_documents: List of context documents to evaluate against
            mode: The evaluation mode (MODE_JUDGE or MODE_EXPLAIN)
            
        Returns:
            str: The raw response from the LLM
            
        Raises:
            ValueError: If no context documents are provided
        """
        if len(context_documents) == 0:
            raise ValueError("No context documents provided")
        context_list = [doc.data for doc in context_documents]
        context_section = "\n* " + "\n* ".join(context_list)
        prompt = prompts[mode].format(context_section=context_section, response=response)
        self.last_prompt = prompt
        return self.llm.generate_response(prompt, "gpt-4o-mini")
        
    def judge(self, response: str, context_documents: list[Document]) -> JudgeResult:
        """
        Evaluate whether a generated response is supported by the context documents.
        
        This method uses an LLM to determine if the response contains factual claims
        that are not explicitly stated in the provided context documents.
        
        Args:
            response: The generated response to evaluate
            context_documents: List of context documents to evaluate against
            
        Returns:
            JudgeResult: 
                - TRUE if the response is supported by the context
                - FALSE if the response contains unsupported claims
                - MAYBE if the evaluation is ambiguous
                
        Raises:
            ValueError: If no context documents are provided
        """
        response = self._judge(response, context_documents, mode=MODE_JUDGE)
        self.last_result = response.strip().lower()
        if "false" in self.last_result and "true" in self.last_result:
            return JudgeResult.MAYBE
        elif "false" in self.last_result:
            return JudgeResult.FALSE
        elif "true" in self.last_result:
            return JudgeResult.TRUE
        else:
            return JudgeResult.MAYBE
        
    def explain(self, response: str, context_documents: list[Document]) -> str:
        """
        Provide a detailed explanation of whether a response is supported by context.
        
        This method uses an LLM to generate a detailed explanation of the evaluation,
        including identification of any unsupported (hallucinated) parts.
        
        Args:
            response: The generated response to evaluate
            context_documents: List of context documents to evaluate against
            
        Returns:
            str: Detailed explanation of the evaluation
            
        Raises:
            ValueError: If no context documents are provided
        """
        explanation = self._judge(response, context_documents, mode=MODE_EXPLAIN)
        self.last_result = explanation.strip()
        return self.last_result
    
    def judge_rerank(self):
        pass

