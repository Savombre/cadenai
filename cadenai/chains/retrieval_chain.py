import json
from . import LLMChain
from ..llm.openai import ChatOpenAI
from ..prompt_manager.template import ChatPromptTemplate
from ..prompt_manager.prompt_list import RETRIEVAL_PROMPT, RETRIEVAL_PROMPT_WITH_METADATA
from ..vectorization.vector_db import VectorDB

class RetrievalChain(LLMChain) : 

    def __init__(self,
                 llm : ChatOpenAI,
                 vector_db : VectorDB,
                 identity : str = "Nice bot created by Cadenai",
                 language : str = "English",
                 include_metadata : bool = False
                 ) -> None :

        self.identity = identity
        self.language = language
        self.vector_db = vector_db
        self.llm = llm
        self.include_metadata = include_metadata
        llm.temperature = 0

        if not self.llm._prompt_syntax == "openai" : 
            raise ValueError("Invalid LLM, only works with ChatOpenAI currently")

        prompt_template = ChatPromptTemplate.from_messages(
                    input_variables=["identity","language","knowledge","user_input"],
                    messages=[
                        ("system", RETRIEVAL_PROMPT),
                        ("human", "{user_input}"),
                        ]
                    )
        
        if self.include_metadata : 
            prompt_template.messages_template[0].content = RETRIEVAL_PROMPT_WITH_METADATA

        super().__init__(prompt_template=prompt_template,llm = llm, max_tokens = 2500)

    def run(self, user_input : str, stream : bool = False) : 

        knowledge = self._retrieve_knowledge_from_vector_db(user_input)

        return super().run(identity=self.identity, language=self.language, knowledge=knowledge, user_input=user_input, stream=stream)
    
    def _retrieve_knowledge_from_vector_db(self, user_input : str, use_metadata : bool = False) : 

        if self.include_metadata :
            brut_knowledge = self.vector_db.similarity_search(query=user_input, limit=5, show_metadata=True)
            knowledge = ""
            for line in brut_knowledge:
                knowledge += json.dumps(line, indent=4) + "\n"

        else : 
            knowledge = "\n".join(self.vector_db.similarity_search(query=user_input, limit=5, show_metadata=False))
        
        return knowledge
