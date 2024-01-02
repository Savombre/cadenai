import pytest
from cadenai.document.file_handler import DocumentHandler
from cadenai.vectorization.vector_db import Qdrant, QdrantManager
from cadenai.vectorization.embeddings import OpenAIEmbeddings
from qdrant_client.http.models import ScoredPoint

@pytest.fixture
def mock_client(mocker):
    client = mocker.Mock()
    client.search.return_value = [
        mocker.MagicMock(payload={"text": "result 1", "other_data": "data 1"}),
        mocker.MagicMock(payload={"text": "result 2", "other_data": "data 2"})
    ]
    return client

@pytest.fixture
def mock_embedder(mocker):
    embedder = mocker.Mock(spec=OpenAIEmbeddings)
    embedder.dimension = 1536  # Configure the mock to return an integer value for dimension
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder

@pytest.fixture
def qdrant_instance(mock_embedder):
    qdrant = Qdrant(
        location="localhost",
        port=1234,
        collection_name="test_collection",
        embedder=mock_embedder,
    )
    return qdrant

def test_len(qdrant_instance,mocker) : 
    mocker.patch('cadenai.vectorization.vector_db.QdrantClient.count', return_value=mocker.MagicMock(count=42))

    result = len(qdrant_instance)
    assert result == 42
    qdrant_instance.client.count.assert_called_with(collection_name="test_collection")

def test_add_documents(mocker, mock_client, mock_embedder, qdrant_instance):
    mock_documents = [DocumentHandler(page_content="test content", metadata={"key": "value"})]
    mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    
    qdrant_instance.client = mock_client
    mocker.patch('cadenai.vectorization.vector_db.Qdrant.__len__', return_value=42)
    mocker.patch.object(qdrant_instance, '_prepare_payloads', return_value=[{"text": "test content", "key": "value"}])

    qdrant_instance.add_documents(mock_documents)

    qdrant_instance._prepare_payloads.assert_called_once_with(mock_documents)
    mock_embedder.embed_documents.assert_called_once_with(documents=mock_documents, loading_bar=False)
    mock_client.upload_records.assert_called_once()

def create_collection(mock_client, qdrant_instance):
    
    qdrant_instance.client = mock_client
    qdrant_instance.create_collection()

    # Verify if embedder and client methods were called as expected
    mock_client.recreate_collection.assert_called_once()

def test_create_from_documents(mocker, qdrant_instance):

    mock_documents = [DocumentHandler(page_content="test content",metadata={"pet foireux": "caca"})]
    qdrant_instance.create_collection = mocker.MagicMock()
    qdrant_instance.add_documents = mocker.MagicMock()

    qdrant_instance.create_from_documents(mock_documents)

    # Verify if the add_documents and _create_collection methods were called as expected
    qdrant_instance.create_collection.assert_called_once()
    qdrant_instance.add_documents.assert_called_once_with(documents=mock_documents, loading_bar=True)

def test_delete_collection(mocker, qdrant_instance, mock_client) : 
    mocker.patch.object(qdrant_instance, 'client', mock_client)
    qdrant_instance.delete_collection()
    mock_client.delete_collection.assert_called_once_with(collection_name="test_collection")

def test_delete_all_collections(mocker, mock_client):
      # Simuler la réponse de get_collections
    collection1 = mocker.MagicMock()
    collection1.name = "collection1"
    collection2 = mocker.MagicMock()
    collection2.name = "collection2"
    mock_collections = [collection1, collection2]
    mock_client.get_collections.return_value = mocker.MagicMock(collections=mock_collections)

    manager = QdrantManager(location="localhost", port=1234, client=mock_client)

    manager.delete_all_collections()

    # Vérifier que delete_collection a été appelé pour chaque collection
    assert mock_client.delete_collection.call_count == len(mock_collections)
    mock_client.delete_collection.assert_any_call(collection_name="collection1")
    mock_client.delete_collection.assert_any_call(collection_name="collection2")

def test_list_all_collections(mocker, mock_client):
    # Simuler la réponse de get_collections
    collection1 = mocker.MagicMock()
    collection1.name = "collection1"
    collection2 = mocker.MagicMock()
    collection2.name = "collection2"
    mock_collections = [collection1, collection2]
    mock_client.get_collections.return_value = mocker.MagicMock(collections=mock_collections)

    manager = QdrantManager(location="localhost", port=1234, client=mock_client)

    result = manager.list_all_collections()

    # Vérifier si le résultat correspond à la liste des noms de collections
    assert result == ["collection1", "collection2"]

    # Vérifier si la méthode get_collections a été appelée correctement
    mock_client.get_collections.assert_called_once()

def test_similarity_search(mocker, qdrant_instance, mock_client, mock_embedder):
    query = "test query"
    limit = 10

    mocker.patch.object(qdrant_instance, 'client', mock_client)

    # Test sans métadonnées
    results_without_metadata = qdrant_instance.similarity_search(query=query, limit=limit)
    mock_embedder.embed_query.assert_called_once_with(query)
    mock_client.search.assert_called_once_with(
        collection_name=qdrant_instance.collection_name,
        query_vector=[0.1, 0.2, 0.3],
        limit=limit
    )
    assert results_without_metadata == ["result 1", "result 2"]

    # Test avec métadonnées
    results_with_metadata = qdrant_instance.similarity_search(query=query, limit=limit, show_metadata=True)
    assert results_with_metadata == [
        {"text": "result 1", "other_data": "data 1"}, 
        {"text": "result 2", "other_data": "data 2"}
    ]

def test_similarity_search_with_scores(mock_client, mock_embedder, qdrant_instance):
    mock_query = "test query"
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_client.search.return_value = [ScoredPoint(id=0, version=0 ,vector=None, payload={"text": "test content", "extra info" : "scoop de zinzin"}, score=1.0)]

    qdrant_instance.client = mock_client

    results = qdrant_instance.similarity_search_with_scores(mock_query, limit=10)

    # Verify results and if embedder and client methods were called
    assert results == [["test content", 1.0]]
    mock_embedder.embed_query.assert_called_once_with(mock_query)
    mock_client.search.assert_called_once()