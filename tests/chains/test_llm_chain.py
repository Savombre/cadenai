import pytest
from cadenai.chains import LLMChain
from cadenai.prompt_manager.template import ChatPromptTemplate

# Fixture pour mock ChatOpenAI
@pytest.fixture
def mock_llm(mocker):
    return mocker.Mock()

# Fixture pour mock BasePromptTemplate
@pytest.fixture
def template_fixture(mocker):
    chat_template = ChatPromptTemplate.from_messages(input_variables=["name","greeting"],messages=[
    ("system", "You are a weird AI bot. Your name is {name}."),
    ("human", "{greeting}"),
    ])
    mock = mocker.Mock()
    mock.format.return_value = "formatted_prompt"
    return chat_template

# Fixture pour créer une instance de LLMChain
@pytest.fixture
def chain(mock_llm, template_fixture):
    return LLMChain(llm=mock_llm, prompt_template=template_fixture)

def test_run(chain, mock_llm, template_fixture, mocker):
    # Mock de la réponse de ChatOpenAI
    mock_llm.get_completion.return_value = "Salut petit coquinou"

    #mocker.patch('cadenai.document.file_handler.PdfReader', return_value=mock_pdf_reader)
    
    response = chain.run(name="Minou", greeting="Coucou Minou")
    
    # Vérifier que le format a été appelé correctement
    template_fixture.format.assert_called_once_with(name="Minou", greeting="Coucou Minou", openai_format=True)
    
    # Vérifier que get_completion a été appelé correctement
    mock_llm.get_completion.assert_called_once_with(prompt=template_fixture.format(name="Minou", greeting="Coucou Minou", openai_format=True), max_tokens=256, stream=False)
    
    # Vérifier que la réponse est correcte
    assert response == "Salut petit coquinou"

def test_run_stream(chain, mock_llm, template_fixture):
    # Mock de la réponse de ChatOpenAI
    mock_llm.get_completion.return_value = "Salut petit coquinou"
    
    response = chain.run(name="Minou", greeting="Coucou Minou", stream=True)
    
    # Vérifier que le format a été appelé correctement
    template_fixture.format.assert_called_once_with(name="Minou", greeting="Coucou Minou", openai_format=True)
    
    # Vérifier que get_completion a été appelé correctement
    mock_llm.get_completion.assert_called_once_with(prompt=template_fixture.format(name="Minou", greeting="Coucou Minou", openai_format=True), max_tokens=256, stream=True)
    
    # Vérifier que la réponse est correcte
    assert response == "Salut petit coquinou"

def test_multiple_runs(chain, mocker, mock_llm, template_fixture):
    # Mock de la réponse de ChatOpenAI pour plusieurs appels
    mock_llm.get_completion.side_effect = ["response_1", "response_2"]
    
    responses = chain.multiple_runs([{"name": "AI1", "greeting": "Hello1"}, {"name": "AI2", "greeting": "Hello2"}])
    
    # Vérifier que le format a été appelé pour chaque entrée
    format_calls = [mocker.call(name="AI1", greeting="Hello1", openai_format=True), 
             mocker.call(name="AI2", greeting="Hello2", openai_format=True)]
    template_fixture.format.assert_has_calls(format_calls, any_order=True)
    
    formatted_prompt_1 = template_fixture.format(name="AI1", greeting="Hello1", openai_format=True)
    formatted_prompt_2 = template_fixture.format(name="AI2", greeting="Hello2", openai_format=True)

    # Vérifier que get_completion a été appelé pour chaque entrée formatée
    get_completion_calls = [mocker.call(prompt=formatted_prompt_1, max_tokens=256, stream=False), mocker.call(prompt=formatted_prompt_2, max_tokens=256, stream=False)]
    mock_llm.get_completion.assert_has_calls(get_completion_calls, any_order=True)
    
    # Vérifier que les réponses sont correctes
    assert responses == ["response_1", "response_2"]

def test_multiple_runs_stream(chain, mocker, mock_llm, template_fixture):
    # Mock de la réponse de ChatOpenAI pour plusieurs appels
    mock_llm.get_completion.side_effect = ["response_1", "response_2"]
    
    responses = chain.multiple_runs([{"name": "AI1", "greeting": "Hello1"}, {"name": "AI2", "greeting": "Hello2"}], stream=True)
    
    # Vérifier que le format a été appelé pour chaque entrée
    format_calls = [mocker.call(name="AI1", greeting="Hello1", openai_format=True), 
             mocker.call(name="AI2", greeting="Hello2", openai_format=True)]
    template_fixture.format.assert_has_calls(format_calls, any_order=True)

    formatted_prompt_1 = template_fixture.format(name="AI1", greeting="Hello1", openai_format=True)
    formatted_prompt_2 = template_fixture.format(name="AI2", greeting="Hello2", openai_format=True)
    
    # Vérifier que get_completion a été appelé pour chaque entrée formatée
    get_completion_calls = [mocker.call(prompt=formatted_prompt_1, max_tokens=256, stream=True), mocker.call(prompt=formatted_prompt_2, max_tokens=256, stream=True)]
    mock_llm.get_completion.assert_has_calls(get_completion_calls, any_order=True)
    
    # Vérifier que les réponses sont correctes
    assert responses == ["response_1", "response_2"]
