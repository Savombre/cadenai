from ...schema import LLM


from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
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

    #@retry(wait=wait_exponential(multiplier=1, min=2, max=4))
    def get_completion(self, prompt : List, max_tokens : int = 500, stream : bool = False) -> str : 

        new_prompt = self._dirty_prompt_formatting(prompt)

        if stream: 
            return self._get_completion_stream(prompt=new_prompt, max_tokens=max_tokens)
        
        else: 
            return self._get_completion_without_stream(prompt=new_prompt, max_tokens=max_tokens) 
        

    # chat_prompt = [{'role': 'system', 'content': 'You are an evil AI bot. Your name is Minou.'}, {'role': 'user', 'content': 'Hello, how are you doing?'}, {'role': 'assistant', 'content': "I'm doing well, thanks!"}, {'role': 'user', 'content': 'What is your name ?'}]
    # llm.get_completion(prompt=chat_prompt)

    def _dirty_prompt_formatting(self, prompt : List) -> List:

        new_prompt = []
        for message in prompt : 
            new_prompt.append(ChatMessage(role=message["role"],content=message["content"]))

        return new_prompt
    
    def _get_completion_without_stream(self, prompt : List, max_tokens : int = 2500) -> str:
        completion = self.client.chat(
        model=self.model,
        temperature = self.temperature,
        messages=prompt,
        #max_tokens=max_tokens,
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

