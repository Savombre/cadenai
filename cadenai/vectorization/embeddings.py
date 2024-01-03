from typing import List, Union

from tenacity import retry

from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, wait_exponential
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from ..schema import DocumentHandler, Embeddings

class OpenAIEmbeddings(Embeddings) : 

    def __init__(self, model : str = "text-embedding-ada-002"):

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dimension = 1536

    def embed_query(self, text : Union[str,DocumentHandler]) -> List[float]:

        if isinstance(text,DocumentHandler) :
            text = text.page_content

        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )

        return response.data[0].embedding
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_with_retry(self, text) -> List[float]:
        return self.embed_query(text=text)

    def embed_documents(self, documents: List[DocumentHandler], loading_bar : bool = True ) -> List[List[float]]:
        '''Embed documents'''

        embeddings: List[List[float]] = []
        if loading_bar : 
            documents = tqdm(documents, desc="Embedding documents")

        for text in documents : 
            if isinstance(text,DocumentHandler) :
                text = text.page_content
            response = self.embed_with_retry(text=text)
            embeddings.append(response)

        return embeddings
        