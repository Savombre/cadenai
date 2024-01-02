from typing import List,Any

import tiktoken

from .schema import Encoder

    
    
class OpenAIEncoder(Encoder) : 

    encoder : Any 

    def __init__(self, model_name="cl100k_base") :
        super().__init__(model_name=model_name)
        self.encoder = tiktoken.get_encoding(self.model_name)

    def encode_a_string(self, text : str) -> List[int]:
        encoded_text = self.encoder.encode(text)
        return encoded_text
    

    def decode_a_string(self,encoded_text : List[int]) -> str: 
        decoded_text = self.encoder.decode(encoded_text)
        return decoded_text
                



