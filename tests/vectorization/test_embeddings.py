import pytest
import openai
from cadenai.vectorization.embeddings import OpenAIEmbeddings
from cadenai.document.file_handler import DocumentHandler

@pytest.fixture
def mock_document():
    mock = DocumentHandler(page_content="Sample content for testing")
    return mock

@pytest.fixture
def mock_openai_client(mocker):
    mock_client = mocker.Mock()
    mock_client.embeddings.create = mocker.Mock()
    return mock_client

def test_embed_query_with_string(mocker, mock_openai_client):

    embedder = OpenAIEmbeddings()
    embedder.client = mock_openai_client

    fake_embedding = [0.1, 0.2, 0.3]
    mock_openai_client.embeddings.create.return_value = mocker.MagicMock(data=[mocker.MagicMock(embedding=fake_embedding)])
    
    result = embedder.embed_query("Test text")

    assert result == [0.1, 0.2, 0.3]
    mock_openai_client.embeddings.create.assert_called_once_with(input="Test text", model="text-embedding-ada-002")

def test_embed_query_with_documenthandler(mocker, mock_openai_client, mock_document):

    embedder = OpenAIEmbeddings()
    embedder.client = mock_openai_client

    fake_embedding = [0.4, 0.5, 0.6]
    mock_openai_client.embeddings.create.return_value = mocker.MagicMock(data=[mocker.MagicMock(embedding=fake_embedding)])

    result = embedder.embed_query(mock_document)

    assert result == [0.4, 0.5, 0.6]
    mock_openai_client.embeddings.create.assert_called_once_with(input=mock_document.page_content, model="text-embedding-ada-002")

def test_embed_with_retry(mocker, mock_openai_client):
    
    embedder = OpenAIEmbeddings()
    embedder.client = mock_openai_client

    fake_embedding = [0.7, 0.8, 0.9]
    mock_openai_client.embeddings.create.return_value = mocker.MagicMock(data=[mocker.MagicMock(embedding=fake_embedding)])

    result = embedder.embed_with_retry("Retry test text")

    assert result == [0.7, 0.8, 0.9]
    mock_openai_client.embeddings.create.assert_called_once_with(input="Retry test text", model="text-embedding-ada-002")

def test_embed_with_retry_retries_on_failure(mocker, mock_openai_client):

    embedder = OpenAIEmbeddings()
    embedder.client = mock_openai_client

    fake_embedding = [0.7, 0.8, 0.9]
    # Mock responses: first call raises an exception, second call returns a mock response
    mock_responses = [mock_openai_client.error.Timeout, mocker.MagicMock(data=[mocker.MagicMock(embedding=fake_embedding)])]

    mock_call = mocker.patch.object(mock_openai_client.embeddings, "create", side_effect=mock_responses)

    # Call embed_with_retry
    result = embedder.embed_with_retry("Retry test text")

    # Check that mock was called twice (i.e., it retried after the exception)
    assert mock_call.call_count == 2

    # Check that the result is from the second mock response
    assert result == [0.7, 0.8, 0.9]

def test_embed_documents(mocker, mock_openai_client, mock_document):

    embedder = OpenAIEmbeddings()
    embedder.client = mock_openai_client

    fake_embedding = [0.7, 0.8, 0.9]
    mock_openai_client.embeddings.create.return_value = mocker.MagicMock(data=[mocker.MagicMock(embedding=fake_embedding)])

    docs = [mock_document, mock_document]
    results = embedder.embed_documents(docs)

    assert results == [[0.7, 0.8, 0.9], [0.7, 0.8, 0.9]]