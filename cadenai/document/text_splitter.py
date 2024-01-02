from enum import Enum
from typing import List, Union, Pattern
import tiktoken
import re #to use regex
import json

from ..schema import DocumentHandler, TextSplitter
from .. import llm
from ..prompt_manager.template import ChatPromptTemplate
from ..prompt_manager.prompt_list import LLMSPLITTER_PROMPT
from ..chains import LLMChain

class ChunkType(Enum):
    CHARACTER = ("character","characters")
    TOKEN = ("token","tokens")
    WORD = ("word","words")

    @classmethod
    def from_str(cls, label: str):
        for item in cls:
            if label in item.value:
                return item
        raise ValueError(f"'{label}' is not a valid ChunkType")

class SizeSplitter(TextSplitter):
    
    def __init__(self,
                chunk_type : str = "characters",
                chunk_size: int = 1000, 
                chunk_overlap: int = 100
                ):
    
        self.chunk_type = ChunkType.from_str(chunk_type)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def _split_text_str(self, input_data : str) -> List[DocumentHandler]:
        return self._split_by_chunk_types(text=input_data)

    def _split_by_chunk_types(self, text : str, model_name : str = "cl100k_base") :

        match self.chunk_type :
            case ChunkType.CHARACTER :
                splitted_text = self._split_a_chunk(text)

            case ChunkType.TOKEN :
                encoding = tiktoken.get_encoding(model_name)
                encoded_text = encoding.encode(text)
                splitted_encoded_text = self._split_a_chunk(encoded_text)
                splitted_text = [encoding.decode(chunk) for chunk in splitted_encoded_text]
                
            case ChunkType.WORD :
                words = text.split()
                splitted_text = self._split_a_chunk(words)
                splitted_text = self._word_list_to_sentence_list(splitted_text)

        return [DocumentHandler(page_content=text) for text in splitted_text]

    def _split_a_chunk(self,text:str)  :

        splitted_text = []
        start = 0
        end = self.chunk_size
        step = self.chunk_size - self.chunk_overlap

        while end <= len(text) :
            splitted_text.append(text[start:end])      
            start+=step
            end+=step

        if  start + self.chunk_overlap < len(text) :
            splitted_text.append(text[start:])
        
        return splitted_text
    
    def _word_list_to_sentence_list(self, word_list : List[str]) -> str :
        sentence_list = []
        for words in word_list :
            sentence_list.append(" ".join(words))
        return sentence_list

class SeparatorSplitter(TextSplitter):

    def __init__(self, 
                separator: Union[str, Pattern] = "\n", 
                is_separator_regex: bool = False
                ):
    
        self.separator = separator
        self.is_separator_regex = is_separator_regex

    def _split_text_str(self, input_data : str) -> List[DocumentHandler]:
        return self._split_with_separator(text=input_data)
    
    def _split_with_separator(self, text : str) -> List[DocumentHandler] : 

        if self.is_separator_regex:
            sections = re.split(self.separator, text)
        else:
            sections = text.split(self.separator)

        splitted_text = []
        print(sections)
        for section in sections:
            print(section)
            # Si la section est vide, on ne l'ajoute pas
            if not section:
                continue
            splitted_text.append(DocumentHandler(page_content=section))

        return splitted_text

class LLMSplitter(TextSplitter,LLMChain):
    
    def __init__(self,
                 document_source : str = "pdf",
                 document_context : str = "from a random file on my computer",
                 llm = llm.ChatOpenAI(model="gpt-4")) : 
        
                self.document_source = document_source
                self.document_context = document_context

                llm.temperature = 0

                prompt_template = ChatPromptTemplate.from_messages(
                    input_variables=["document_source","document_context","document_text"],
                    messages=[
                        ("system", LLMSPLITTER_PROMPT),
                        ("human", "{document_text}"),
                        ]
                    )
                super().__init__(prompt_template=prompt_template,llm = llm, max_tokens = 2500)

    def _split_text_str(self, input_data : str) -> List[DocumentHandler]:

        llm_response = self.run(document_source=self.document_source,document_context=self.document_context,document_text=input_data)
        json_loaded = json.loads(llm_response)
        documents = [DocumentHandler(**item) for item in json_loaded]
        return documents
