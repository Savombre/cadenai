from typing import List, Any

from qdrant_client import QdrantClient, models

from ..schema import VectorDB, Embeddings, DocumentHandler

class QdrantManager() : 

    def __init__(self,
                 location : str,
                 port : int,
                 client : Any = None
                 ): 
        
        self.location = location
        self.port = port

        self.client = client if client else QdrantClient(location=self.location, port=self.port)

    def list_all_collections(self) : 
        return [collection.name for collection in self.client.get_collections().collections]

    def delete_all_collections(self) : 
        for collection_name in self.list_all_collections() :
            self.client.delete_collection(collection_name=collection_name)

class Qdrant(VectorDB): 

    """
    Make sure to have a docker running with the qdrant db
    """

    def __init__(self,
        location : str, 
        port : int,
        collection_name : str,
        embedder : Embeddings
    ): 
        self.location = location
        self.port = port
        self.client = QdrantClient(
            location=self.location,
            port=self.port
        )
        self.collection_name = collection_name
        self.embedder = embedder

    def __len__(self) -> int:
        return self.client.count(collection_name=self.collection_name).count
    
    def add_documents(self,documents : List[DocumentHandler], loading_bar : bool = False) : 

        documents_embedded = self.embedder.embed_documents(documents=documents,loading_bar=loading_bar)
        payloads = self._prepare_payloads(documents)
        vector_list = self._prepare_vector_list(documents_embedded, payloads)

        operation_info = self.client.upload_records(
            collection_name=self.collection_name,
            wait=True,
            records = vector_list
        )
        return operation_info
    
    def create_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension, 
                distance=models.Distance.COSINE
            )
        )

    def create_from_documents(self, documents: List[DocumentHandler], loading_bar : bool = True):
        self.create_collection()
        return self.add_documents(documents=documents, loading_bar=loading_bar)
    
    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)
    
    def similarity_search(self, query : str, limit : int, show_metadata : bool = False) -> List[str] :
            
        query_vector = self.embedder.embed_query(query)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        if show_metadata : 
            return [result.payload for result in search_result]
        else : 
            return [result.payload["text"] for result in search_result]
    
    def similarity_search_with_scores(self, query : str, limit : int) -> List :

        query_vector = self.embedder.embed_query(query)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )

        output = []
        for result in search_result : 
            output.append([result.payload["text"],result.score])

        return output

    def _prepare_payloads(self, documents : List[DocumentHandler]) -> List[dict]: 
        
        payloads = []
        for document in documents : 
            if isinstance(document, DocumentHandler) : 
                text = document.page_content
                metadata = document.metadata.copy()
                metadata["text"] = text
                payloads.append(metadata)
            else : 
                raise Exception("Documents must be of type DocumentHandler")
            
        return payloads
    
    def _prepare_vector_list(self, documents_embedded : List[List[float]], payloads : List[dict]):
                
        vector_list = []

        i=len(self)
        for vector, payload in zip(documents_embedded, payloads) :
            vector_list.append(models.Record(
                id=i,
                vector=vector,
                payload=payload
            ))
            i+=1

        return vector_list