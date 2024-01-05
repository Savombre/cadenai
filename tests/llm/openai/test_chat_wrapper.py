import pytest
from cadenai.llm.openai import ChatOpenAI

@pytest.fixture
def mock_openai_stream(mocker):
    mock_client = mocker.Mock()
    # Simulez le comportement de la méthode chat.completions.create
    # Vous devez retourner un itérateur ou un générateur ici
    mock_completion = iter([
        mocker.MagicMock(choices=[mocker.MagicMock(delta=mocker.MagicMock(content="token1"))]),
        mocker.MagicMock(choices=[mocker.MagicMock(delta=mocker.MagicMock(content="token2"))])
    ])
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client

@pytest.fixture
def mock_openai(mocker):
    mock_client = mocker.Mock()
    # Simulez le comportement de la méthode chat.completions.create pour un appel non-stream
    mock_completion = mocker.MagicMock(choices=[mocker.MagicMock(message=mocker.MagicMock(content="Réponse simulée"))])
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client

def test_get_completion_stream(mock_openai_stream):
    chat_ai = ChatOpenAI(model="gpt-4")
    chat_ai.client = mock_openai_stream  # Injectez le mock client

    prompt = [{"role": "user", "content": "Test"}]
    generator = chat_ai.get_completion(prompt, max_tokens=50,stream=True)

    # Vérifiez si les tokens sont générés correctement
    tokens = list(generator)
    assert tokens == ["token1", "token2"]

    # Vérifiez si la méthode create a été appelée correctement
    mock_openai_stream.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        temperature=0.7,
        messages=prompt,
        max_tokens=50,
        stream=True
    )

def test_get_completion(mock_openai):

    chat_ai = ChatOpenAI(model="gpt-4")
    chat_ai.client = mock_openai  # Injectez le mock client

    prompt = [{"role": "user", "content": "Test"}]
    result = chat_ai.get_completion(prompt, max_tokens=50, stream=False)

    # Vérifiez si le résultat correspond à ce qui est attendu
    assert result == "Réponse simulée"

    # Vérifiez si la méthode create a été appelée correctement
    mock_openai.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        temperature=0.7,
        messages=prompt,
        max_tokens=50,
    )

