from ..schema import BasePromptTemplate, LLM

from typing import List

class LLMChain():

    def __init__(self, 
                 llm : LLM, 
                 prompt_template : BasePromptTemplate,
                 max_tokens : int = 256,
                 ) -> None:
        
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens

    def run(self, stream : bool = False, **kwargs) : 
        prompt = self.prompt_template.format(syntax=self.llm._prompt_syntax,**kwargs)
        return self.llm.get_completion(prompt=prompt, max_tokens=self.max_tokens, stream=stream)

    def multiple_runs(self, input_list : List, stream : bool = False) : 
        output_list = []
        for input_variables in input_list : 
            output_list.append(self.run(**input_variables, stream=stream))
        return output_list


