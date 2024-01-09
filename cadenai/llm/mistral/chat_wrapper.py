from ...schema import LLM, BasePromptTemplate


from mistralai.client import MistralClient
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from typing import List
from tenacity import retry, wait_exponential


class ChatMistral(LLM):
    def __init__(self,
                 model : str = "mistral-tiny",
                 temperature : float = 0.7
                 ):
        self.model = model
        self.temperature = temperature #Can't go upper than 1
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
        self._prompt_syntax = "mistral"

    @retry(wait=wait_exponential(multiplier=1, min=2, max=4))
    def get_completion(self, prompt : List, max_tokens : int = 500, stream : bool = False) -> str : 
        
        if stream: 
            return self._get_completion_stream(prompt=prompt, max_tokens=max_tokens)
        
        else: 
            return self._get_completion_without_stream(prompt=prompt, max_tokens=max_tokens) 
    
    def _get_completion_without_stream(self, prompt : List, max_tokens : int = 2500) -> str:
        completion = self.client.chat(
        model=self.model,
        temperature = self.temperature,
        messages=prompt,
        max_tokens=max_tokens,
        )

        return completion.choices[0].message.content
    
    def _get_completion_stream(self, prompt : List, max_tokens : int = 2500) -> str:
        
        completion = self.client.chat_stream(
        model=self.model,
        temperature = self.temperature,
        messages=prompt,
        max_tokens=max_tokens,
        )

        for chunk in completion:
            yield chunk.choices[0].delta.content

