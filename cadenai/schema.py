from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Iterator
from tqdm import tqdm
from pydantic import BaseModel, Field
import json

class BasePromptTemplate(ABC) : 

    @abstractmethod
    def format(self, **kwargs) -> str : 
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of tokens in the prompt template"""
        pass

class DocumentHandler(BaseModel) : 

    page_content : str
    metadata: Optional[Dict[str, Union[int,str]]] = Field(default_factory=dict)

    def save(self, path : str) -> None : 
        try:
            with open(path, 'w', encoding='utf-8') as file:
                # Convertit l'instance en dictionnaire et l'écrit dans un fichier JSON
                json.dump(self.model_dump(), file, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"Erreur lors de l'enregistrement dans le fichier : {e}")

class Encoder(ABC):

    model_name : str

    def __init__(self, model_name : str) : 
        self.model_name = model_name

    @abstractmethod
    def encode_a_string(self, text : str) -> List[int]: 
        pass

    @abstractmethod
    def decode_a_string(self, encoded_text : List[int]) -> str:
        pass

    # @abstractmethod
    # def encode_a_document(self, text : str) -> List[int]: 
    #     pass

    # @abstractmethod
    # def decode_a_document(self, encoded_text : List[int]) -> str:
    #     pass

class Embeddings(ABC): 
    """Interface for embedding models."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

class Loader(ABC): 

    def load(self) -> List[DocumentHandler] : 
        return list(self.lazy_load())

    @abstractmethod
    def lazy_load(self) -> Iterator[DocumentHandler] : 
        pass

    @abstractmethod
    def __len__(self) -> int : 
        pass

class TextSplitter(ABC):

    def split_text(self, input_data : Union[DocumentHandler,List[DocumentHandler], str, List[str], Loader], loading_bar : bool = True ) -> List[DocumentHandler]:

        if isinstance(input_data, Loader):
            return self._split_text_loader(input_data=input_data,loading_bar=loading_bar)

        elif isinstance(input_data, list) and all(isinstance(item, DocumentHandler) for item in input_data): 
            return self._split_text_DocumentHandler_list(input_data=input_data, loading_bar=loading_bar)

        elif isinstance(input_data, DocumentHandler):
            return self._split_text_DocumentHandler(input_data=input_data)
            
        elif isinstance(input_data, str):
            return self._split_text_str(input_data=input_data)
            
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            return self._split_text_str_list(input_data=input_data, loading_bar=loading_bar)

        else : #Or raise an exception
            raise TypeError(f"Unsupported input type: {type(input_data).__name__}.")
        
    def _split_text_loader(self, input_data : Loader, loading_bar : bool = True) :
        splitted_text = []
        if loading_bar :
            iterable = tqdm(input_data.lazy_load(), total=len(input_data), desc="Loading and Splitting Documents")
        else :
            iterable = input_data.lazy_load()
        for doc in iterable:
            splitted_text.extend(self._split_text_DocumentHandler(input_data=doc))
        return splitted_text
    
    def _split_text_DocumentHandler_list(self, input_data : List[DocumentHandler], loading_bar : bool = True) :
        splitted_text = []
        if loading_bar :
            iterable = tqdm(input_data, desc="Splitting Documents")
        else :
            iterable = input_data
        for doc in iterable:
            splitted_text.extend(self._split_text_DocumentHandler(input_data=doc))
        return splitted_text
    
    def _split_text_DocumentHandler(self, input_data : DocumentHandler) :
        splitted_documents = self._split_text_str(input_data=input_data.page_content)
        for document in splitted_documents :
            document.metadata = input_data.metadata
        return splitted_documents

    def _split_text_str_list(self, input_data : List[str], loading_bar : bool = True)  :
        splitted_text = []
        if loading_bar :
            iterable = tqdm(input_data, desc="Splitting Texts")
        else :
            iterable = input_data
        for doc in iterable:
            splitted_text.extend(self._split_text_str(input_data=doc))
        return splitted_text

    @abstractmethod
    def _split_text_str(self, input_data : str) -> List[DocumentHandler]:
        pass

class VectorDB(ABC) : 
    
    @abstractmethod
    def add_documents(self, documents : List[DocumentHandler]) : 
        pass

    @abstractmethod
    def create_from_documents(self, document : List[DocumentHandler], embedder : Embeddings):
        pass

    @abstractmethod
    def similarity_search(self, query : str, limit : int) -> List[str]:
        pass

    @abstractmethod
    def similarity_search_with_scores(self, query : str, limit : int) -> List :
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass