from pydantic import BaseModel
from typing import List, Tuple
from enum import Enum
import tiktoken
from mistralai.models.chat_completion import ChatMessage as MistralChatMessage

from ..schema import BasePromptTemplate, PromptSyntax

class Role(Enum):
    SYSTEM = ("system","system")
    AI = ("ai","assistant")
    HUMAN = ("human","user")

    @property
    def cadenai(self) -> str : 
        return self.value[0]
    
    @property
    def openai(self) -> str : 
        return self.value[1]
    
    @property
    def mistral(self) -> str : 
        return self.value[1]

    @classmethod
    def from_role_name(cls, role_name):
        for role in cls:
            if role.cadenai == role_name or role.openai == role_name:
                return role
        raise ValueError(f"Invalid role name: {role_name}")

def token_counter(text,model_type="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_type)
    encoded = encoding.encode(text)
    num_tokens = len(encoded)
    return num_tokens

class PromptTemplate(BaseModel,BasePromptTemplate) : 

    input_variables: List[str]
    template : str  #f-string template

    def format(self, **kwargs) -> str : 
        return self.template.format(**kwargs)
    
    def __str__(self) -> str:
        return self.template
    
    def __len__(self) -> str:
        return token_counter(self.template)
    
class MessageTemplate(BaseModel, BasePromptTemplate) :

    role : Role
    content : str

    def format(self, syntax : str, **kwargs) -> str : 
        prompt_syntax = PromptSyntax.from_str(syntax)
        match prompt_syntax : 
            case prompt_syntax.OPENAI : 
                return {"role" : self.role.openai, "content" : self.content.format(**kwargs)}
            case prompt_syntax.MISTRAL :
                return MistralChatMessage(role=self.role.mistral, content=self.content.format(**kwargs))
            case prompt_syntax.CADENAI :
                return (self.role.cadenai,self.content.format(**kwargs))
    
    def __str__(self) -> str:
        return repr((self.role.cadenai,self.content))
    
    def __len__(self) -> int:
        return token_counter(self.content)

class ChatPromptTemplate(BaseModel,BasePromptTemplate) : 

    input_variables: List[str]
    messages_template : List[MessageTemplate]  

    @classmethod
    def from_messages(cls, input_variables : List[str], messages :  List[Tuple]) : 
        instance = cls(input_variables=input_variables, messages_template=[])
        for message in messages : 
            instance.messages_template.append(MessageTemplate(role=Role.from_role_name(message[0]), content=message[1]))
        return instance

    def format(self, syntax : str, **kwargs) -> str : 
        formatted_template = []
        for message_template in self.messages_template :
            formatted_template.append(message_template.format(syntax=syntax,**kwargs))
        return formatted_template
    
    def add_system_message(self, content : str, input_variables : List[str] = None) : 
        if input_variables : 
            self.input_variables.extend(input_variables)
        self.messages_template.append(MessageTemplate(role=Role.SYSTEM, content=content))

    def add_ai_message(self, content : str, input_variables : List[str] = None) : 
        if input_variables : 
            self.input_variables.extend(input_variables)
        self.messages_template.append(MessageTemplate(role=Role.AI, content=content))
    
    def add_human_message(self, content : str, input_variables : List[str] = None) : 
        if input_variables : 
            self.input_variables.extend(input_variables)
        self.messages_template.append(MessageTemplate(role=Role.HUMAN, content=content))
    
    def __str__(self) -> str:
        output = []
        for message_template in self.messages_template:
            output.append((message_template.role.cadenai, message_template.content))
        return repr(output)

    def __len__(self) -> int:
        length = 0
        for message_template in self.messages_template:
            length += len(message_template)
        return length