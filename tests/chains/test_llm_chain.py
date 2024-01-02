from httpx import stream
import pytest
from cadenai.chains import LLMChain
from cadenai.llm import ChatOpenAI
from cadenai.prompt_manager.template import BasePromptTemplate

# Fixture pour mock ChatOpenAI
@pytest.fixture
def mock_llm(mocker):
    return mocker.Mock()

# Fixture pour mock BasePromptTemplate
@pytest.fixture
def mock_template(mocker):
    mock = mocker.Mock()
    mock.format.return_value = "formatted_prompt"
    return mock

# Fixture pour créer une instance de LLMChain
@pytest.fixture
def chain(mock_llm, mock_template):
    return LLMChain(llm=mock_llm, prompt_template=mock_template)

def test_run(chain, mock_llm, mock_template):
    # Mock de la réponse de ChatOpenAI
    mock_llm.get_completion.return_value = "ChatOpenAI_response"
    
    response = chain.run(name="Minou", greeting="Coucou Minou")
    
    # Vérifier que le format a été appelé correctement
    mock_template.format.assert_called_once_with(name="Minou", greeting="Coucou Minou", openai_format=True)
    
    # Vérifier que get_completion a été appelé correctement
    mock_llm.get_completion.assert_called_once_with(prompt="formatted_prompt", max_tokens=256, stream=False)
    
    # Vérifier que la réponse est correcte
    assert response == "ChatOpenAI_response"

def test_run_stream(chain, mock_llm, mock_template):
    # Mock de la réponse de ChatOpenAI
    mock_llm.get_completion.return_value = "ChatOpenAI_response"
    
    response = chain.run(name="Minou", greeting="Coucou Minou", stream=True)
    
    # Vérifier que le format a été appelé correctement
    mock_template.format.assert_called_once_with(name="Minou", greeting="Coucou Minou", openai_format=True)
    
    # Vérifier que get_completion a été appelé correctement
    mock_llm.get_completion.assert_called_once_with(prompt="formatted_prompt", max_tokens=256, stream=True)
    
    # Vérifier que la réponse est correcte
    assert response == "ChatOpenAI_response"

def test_multiple_runs(chain, mocker, mock_llm, mock_template):
    # Mock de la réponse de ChatOpenAI pour plusieurs appels
    mock_llm.get_completion.side_effect = ["response_1", "response_2"]
    
    responses = chain.multiple_runs([{"name": "AI1", "greeting": "Hello1"}, {"name": "AI2", "greeting": "Hello2"}])
    
    # Vérifier que le format a été appelé pour chaque entrée
    calls = [mocker.call(name="AI1", greeting="Hello1", openai_format=True), 
             mocker.call(name="AI2", greeting="Hello2", openai_format=True)]
    mock_template.format.assert_has_calls(calls, any_order=True)
    
    # Vérifier que get_completion a été appelé pour chaque entrée formatée
    calls = [mocker.call(prompt="formatted_prompt", max_tokens=256, stream=False), mocker.call(prompt="formatted_prompt", max_tokens=256, stream=False)]
    mock_llm.get_completion.assert_has_calls(calls, any_order=True)
    
    # Vérifier que les réponses sont correctes
    assert responses == ["response_1", "response_2"]

def test_multiple_runs_stream(chain, mocker, mock_llm, mock_template):
    # Mock de la réponse de ChatOpenAI pour plusieurs appels
    mock_llm.get_completion.side_effect = ["response_1", "response_2"]
    
    responses = chain.multiple_runs([{"name": "AI1", "greeting": "Hello1"}, {"name": "AI2", "greeting": "Hello2"}], stream=True)
    
    # Vérifier que le format a été appelé pour chaque entrée
    calls = [mocker.call(name="AI1", greeting="Hello1", openai_format=True), 
             mocker.call(name="AI2", greeting="Hello2", openai_format=True)]
    mock_template.format.assert_has_calls(calls, any_order=True)
    
    # Vérifier que get_completion a été appelé pour chaque entrée formatée
    calls = [mocker.call(prompt="formatted_prompt", max_tokens=256, stream=True), mocker.call(prompt="formatted_prompt", max_tokens=256, stream=True)]
    mock_llm.get_completion.assert_has_calls(calls, any_order=True)
    
    # Vérifier que les réponses sont correctes
    assert responses == ["response_1", "response_2"]
