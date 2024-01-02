import pytest
import json

from cadenai.chains import RetrievalChain
from cadenai.prompt_manager.prompt_list import RETRIEVAL_PROMPT, RETRIEVAL_PROMPT_WITH_METADATA

@pytest.fixture
def mock_llm(mocker):
    mock = mocker.MagicMock()
    return mock

@pytest.fixture
def mock_vector_db(mocker):
    mock = mocker.MagicMock()
    return mock

@pytest.fixture
def mock_retrieval_chain_without_metadata(mock_llm, mock_vector_db):
    return RetrievalChain(llm=mock_llm, 
                          vector_db=mock_vector_db, 
                          identity="Nice bot created by Cadenai", 
                          language="English",
                          include_metadata=False)

@pytest.fixture
def mock_retrieval_chain_with_metadata(mock_llm, mock_vector_db):
    return RetrievalChain(llm=mock_llm, 
                          vector_db=mock_vector_db, 
                          identity="Nice bot created by Cadenai", 
                          language="English",
                          include_metadata=True)

def test_retrieval_chain_init(mock_llm, mock_vector_db):
    retrieval_chain = RetrievalChain(llm=mock_llm, vector_db=mock_vector_db)
    
    assert retrieval_chain.identity == "Nice bot created by Cadenai"
    assert retrieval_chain.language == "English"
    assert retrieval_chain.vector_db is mock_vector_db
    assert retrieval_chain.llm is mock_llm
    assert retrieval_chain.include_metadata is False

def test_prompt_template_with_metadata(): 
    retrieval_chain_with_metadata = RetrievalChain(llm=mock_llm, vector_db=mock_vector_db, include_metadata=True)
    assert retrieval_chain_with_metadata.prompt_template.messages_template[0].content == RETRIEVAL_PROMPT_WITH_METADATA


def test_prompt_template_without_metadata():
    
    retrieval_chain_without_metadata = RetrievalChain(llm=mock_llm, vector_db=mock_vector_db)
    assert retrieval_chain_without_metadata.prompt_template.messages_template[0].content == RETRIEVAL_PROMPT


def test_retrieval_chain_retrieve_knowledge_from_vector_db_without_metadata(mock_retrieval_chain_without_metadata, mock_vector_db):

    mock_search_results = ["result 1", "result 2"]
    mock_vector_db.similarity_search.return_value = mock_search_results

    # Appel de la méthode à tester
    knowledge = mock_retrieval_chain_without_metadata._retrieve_knowledge_from_vector_db("test query")

    # Vérifiez si le résultat est correctement formaté
    expected_knowledge = "\n".join(mock_search_results)
    assert knowledge == expected_knowledge

    # Vérifiez si similarity_search a été appelée correctement
    mock_vector_db.similarity_search.assert_called_once_with(query="test query", limit=5, show_metadata=False)

def test_retrieval_chain_retrieve_knowledge_from_vector_db_with_metadata(mock_retrieval_chain_with_metadata, mock_vector_db):

    mock_search_results = [
        {"text": "result 1", "metadata": {"key1": "value1"}},
        {"text": "result 2", "metadata": {"key2": "value2"}}
    ]
    mock_vector_db.similarity_search.return_value = mock_search_results

    knowledge = mock_retrieval_chain_with_metadata._retrieve_knowledge_from_vector_db("test query")

    # Vérifiez si le résultat est correctement formaté
    expected_knowledge = "\n".join(json.dumps(line, indent=4) for line in mock_search_results) + "\n"
    assert knowledge == expected_knowledge
    mock_vector_db.similarity_search.assert_called_once_with(query="test query", limit=5, show_metadata=True)

def test_retrieval_chain_run_with_metadata(mocker, mock_llm, mock_vector_db):

    expected_result = "C'est Marseille bébé"

    mock_llm.get_completion.return_value = expected_result
    mock_search_results = [
        {"text": "fact 1", "metadata": {"value": "objective"}},
        {"text": "fact 2", "metadata": {"value": "bullshit"}},
        {"text": "fact 3", "metadata": {"value": "subjective"}},
        {"text": "fact 4", "metadata": {"value": "like this one"}},
        {"text": "fact 5", "metadata": {"value": "totally not a fact"}}
    ]
    mock_vector_db.similarity_search.return_value = mock_search_results

    retrieval_chain_with_metadata = RetrievalChain(llm=mock_llm, vector_db=mock_vector_db, include_metadata=True)
    user_input = "Quel est la capitale de la France ?"

    expected_prompt = retrieval_chain_with_metadata.prompt_template.format(
        identity=retrieval_chain_with_metadata.identity,
        language=retrieval_chain_with_metadata.language,
        knowledge=retrieval_chain_with_metadata._retrieve_knowledge_from_vector_db(user_input), #First similarity_search call
        user_input=user_input,
        openai_format=True
    )

    result = retrieval_chain_with_metadata.run(user_input=user_input) #Second similarity_search call

    expected_calls = [
        mocker.call(query='Quel est la capitale de la France ?', limit=5, show_metadata=True),
        mocker.call(query='Quel est la capitale de la France ?', limit=5, show_metadata=True)
    ]
    mock_vector_db.similarity_search.assert_has_calls(expected_calls)

    assert result == expected_result

    mock_llm.get_completion.assert_called_once_with(
        prompt=expected_prompt,
        max_tokens=retrieval_chain_with_metadata.max_tokens,
        stream=False
    )

def test_retrieval_chain_run_without_metadata(mock_llm, mock_vector_db):

    mock_llm.get_completion.return_value = "C'est Marseille bébé"
    mock_vector_db.similarity_search.return_value = ["fact1", "fact2", "fact3", "fact4", "fact5"]

    retrieval_chain_without_metadata = RetrievalChain(llm=mock_llm, vector_db=mock_vector_db)
    user_input = "Quel est la capitale de la France ?"

    expected_prompt = retrieval_chain_without_metadata.prompt_template.format(
        identity=retrieval_chain_without_metadata.identity,
        language=retrieval_chain_without_metadata.language,
        knowledge="\n".join(["fact1", "fact2", "fact3", "fact4", "fact5"]),
        user_input=user_input,
        openai_format=True
    )

    result = retrieval_chain_without_metadata.run(user_input=user_input)

    mock_vector_db.similarity_search.assert_called_once_with(query=user_input, limit=5, show_metadata=False)
    assert result == "C'est Marseille bébé"

    mock_llm.get_completion.assert_called_once_with(
        prompt=expected_prompt,
        max_tokens=retrieval_chain_without_metadata.max_tokens,
        stream=False
    )

def test_retrieval_chain_run_with_stream(mock_llm, mock_vector_db): 

    mock_llm.get_completion.return_value = "C'est Marseille bébé"
    mock_vector_db.similarity_search.return_value = ["fact1", "fact2", "fact3", "fact4", "fact5"]

    retrieval_chain_without_metadata = RetrievalChain(llm=mock_llm, vector_db=mock_vector_db)
    user_input = "Quel est la capitale de la France ?"

    expected_prompt = retrieval_chain_without_metadata.prompt_template.format(
        identity=retrieval_chain_without_metadata.identity,
        language=retrieval_chain_without_metadata.language,
        knowledge="\n".join(["fact1", "fact2", "fact3", "fact4", "fact5"]),
        user_input=user_input,
        openai_format=True
    )

    result = retrieval_chain_without_metadata.run(user_input=user_input, stream=True)

    mock_vector_db.similarity_search.assert_called_once_with(query=user_input, limit=5, show_metadata=False)
    assert result == "C'est Marseille bébé"

    mock_llm.get_completion.assert_called_once_with(
        prompt=expected_prompt,
        max_tokens=retrieval_chain_without_metadata.max_tokens,
        stream=True
    )


def test_retrieval_chain_run_with_stream_and_metadata(mocker, mock_llm, mock_vector_db):

    expected_result = "C'est Marseille bébé"

    mock_llm.get_completion.return_value = expected_result
    mock_search_results = [
        {"text": "fact 1", "metadata": {"value": "objective"}},
        {"text": "fact 2", "metadata": {"value": "bullshit"}},
        {"text": "fact 3", "metadata": {"value": "subjective"}},
        {"text": "fact 4", "metadata": {"value": "like this one"}},
        {"text": "fact 5", "metadata": {"value": "totally not a fact"}}
    ]
    mock_vector_db.similarity_search.return_value = mock_search_results

    retrieval_chain_with_metadata = RetrievalChain(llm=mock_llm, vector_db=mock_vector_db, include_metadata=True)
    user_input = "Quel est la capitale de la France ?"

    expected_prompt = retrieval_chain_with_metadata.prompt_template.format(
        identity=retrieval_chain_with_metadata.identity,
        language=retrieval_chain_with_metadata.language,
        knowledge=retrieval_chain_with_metadata._retrieve_knowledge_from_vector_db(user_input), #First similarity_search call
        user_input=user_input,
        openai_format=True
    )

    result = retrieval_chain_with_metadata.run(user_input=user_input, stream=True) #Second similarity_search call

    expected_calls = [
        mocker.call(query='Quel est la capitale de la France ?', limit=5, show_metadata=True),
        mocker.call(query='Quel est la capitale de la France ?', limit=5, show_metadata=True)
    ]
    mock_vector_db.similarity_search.assert_has_calls(expected_calls)

    assert result == expected_result

    mock_llm.get_completion.assert_called_once_with(
        prompt=expected_prompt,
        max_tokens=retrieval_chain_with_metadata.max_tokens,
        stream=True
    )