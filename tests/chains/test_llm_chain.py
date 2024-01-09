import pytest
from cadenai.chains import LLMChain
from cadenai.prompt_manager.template import ChatPromptTemplate

# Fixture pour mock ChatOpenAI
@pytest.fixture
def mock_llm(mocker):
    return mocker.Mock()

# Pas utilisé car ça me fait des bugs, obligé de tout mocker
# @pytest.fixture
# def template_fixture():
#     chat_template = ChatPromptTemplate.from_messages(input_variables=["name","greeting"],messages=[
#     ("system", "You are a weird AI bot. Your name is {name}."),
#     ("human", "{greeting}"),
#     ])

#     return chat_template

@pytest.fixture
def mock_template(mocker):
    mock = mocker.Mock()
    mock.format.return_value = "Salut petit coquinou"
    return mock

# Fixture pour créer une instance de LLMChain
@pytest.fixture
def chain(mock_llm, mock_template):
    return LLMChain(llm=mock_llm, prompt_template=mock_template)

@pytest.mark.parametrize("_prompt_syntax",["openai","mistral"])
def test_run(chain, mock_llm, mock_template, _prompt_syntax):
    # Mock de la réponse de ChatOpenAI
    mock_llm.get_completion.return_value = "Salut petit coquinou"
    mock_llm._prompt_syntax = _prompt_syntax


    response = chain.run(name="Minou", greeting="Coucou Minou")
    
    # Vérifier que le format a été appelé correctement
    mock_template.format.assert_called_once_with(syntax=_prompt_syntax, name="Minou", greeting="Coucou Minou")
    
    # Vérifier que get_completion a été appelé correctement
    mock_llm.get_completion.assert_called_once_with(prompt=mock_template.format(name="Minou", greeting="Coucou Minou"), max_tokens=256, stream=False)
    
    # Vérifier que la réponse est correcte
    assert response == "Salut petit coquinou"

@pytest.mark.parametrize("_prompt_syntax",["openai","mistral"])
def test_run_stream(chain, mock_llm, mock_template, _prompt_syntax):
    # Mock de la réponse de ChatOpenAI
    mock_llm.get_completion.return_value = "Salut petit coquinou"
    mock_llm._prompt_syntax = _prompt_syntax
    
    response = chain.run(name="Minou", greeting="Coucou Minou", stream=True)
    
    # Vérifier que le format a été appelé correctement
    mock_template.format.assert_called_once_with(syntax=_prompt_syntax, name="Minou", greeting="Coucou Minou")
    
    # Vérifier que get_completion a été appelé correctement
    mock_llm.get_completion.assert_called_once_with(prompt=mock_template.format(name="Minou", greeting="Coucou Minou"), max_tokens=256, stream=True)
    
    # Vérifier que la réponse est correcte
    assert response == "Salut petit coquinou"

@pytest.mark.parametrize("_prompt_syntax",["openai","mistral"])
def test_multiple_runs(chain, mocker, mock_llm, mock_template, _prompt_syntax):
    # Mock de la réponse de ChatOpenAI pour plusieurs appels
    mock_llm.get_completion.side_effect = ["response_1", "response_2"]
    mock_llm._prompt_syntax = _prompt_syntax
    
    responses = chain.multiple_runs([{"name": "AI1", "greeting": "Hello1"}, {"name": "AI2", "greeting": "Hello2"}])
    
    # Vérifier que le format a été appelé pour chaque entrée
    format_calls = [mocker.call(syntax=_prompt_syntax, name="AI1", greeting="Hello1"), 
             mocker.call(syntax=_prompt_syntax, name="AI2", greeting="Hello2")]
    mock_template.format.assert_has_calls(format_calls, any_order=True)
    
    formatted_prompt_1 = mock_template.format(syntax=_prompt_syntax, name="AI1", greeting="Hello1")
    formatted_prompt_2 = mock_template.format(syntax=_prompt_syntax, name="AI2", greeting="Hello2")

    # Vérifier que get_completion a été appelé pour chaque entrée formatée
    get_completion_calls = [mocker.call(prompt=formatted_prompt_1, max_tokens=256, stream=False), mocker.call(prompt=formatted_prompt_2, max_tokens=256, stream=False)]
    mock_llm.get_completion.assert_has_calls(get_completion_calls, any_order=True)
    
    # Vérifier que les réponses sont correctes
    assert responses == ["response_1", "response_2"]

@pytest.mark.parametrize("_prompt_syntax",["openai","mistral"])
def test_multiple_runs_stream(chain, mocker, mock_llm, mock_template, _prompt_syntax):
    # Mock de la réponse de ChatOpenAI pour plusieurs appels
    mock_llm.get_completion.side_effect = ["response_1", "response_2"]
    mock_llm._prompt_syntax = _prompt_syntax

    responses = chain.multiple_runs([{"name": "AI1", "greeting": "Hello1"}, {"name": "AI2", "greeting": "Hello2"}], stream=True)
    
    # Vérifier que le format a été appelé pour chaque entrée
    format_calls = [mocker.call(syntax=_prompt_syntax, name="AI1", greeting="Hello1"), 
             mocker.call(syntax=_prompt_syntax, name="AI2", greeting="Hello2")]
    mock_template.format.assert_has_calls(format_calls, any_order=True)

    formatted_prompt_1 = mock_template.format(syntax=_prompt_syntax, name="AI1", greeting="Hello1")
    formatted_prompt_2 = mock_template.format(syntax=_prompt_syntax, name="AI2", greeting="Hello2")
    
    # Vérifier que get_completion a été appelé pour chaque entrée formatée
    get_completion_calls = [mocker.call(prompt=formatted_prompt_1, max_tokens=256, stream=True), mocker.call(prompt=formatted_prompt_2, max_tokens=256, stream=True)]
    mock_llm.get_completion.assert_has_calls(get_completion_calls, any_order=True)
    
    # Vérifier que les réponses sont correctes
    assert responses == ["response_1", "response_2"]
