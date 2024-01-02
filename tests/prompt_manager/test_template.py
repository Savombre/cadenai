import openai
import pytest
from cadenai.prompt_manager.template import PromptTemplate, MessageTemplate, ChatPromptTemplate, Role


def test_prompt_template_formatting():
    pt = PromptTemplate(input_variables=["name"], template="Hello, {name}!")
    assert pt.format(name="John") == "Hello, John!"
    assert str(pt) == "Hello, {name}!"

def test_message_template_formatting():
    mt = MessageTemplate(role=Role.AI, content="Hello, {name}!")
    assert mt.format(name="John") == (Role.AI.cadenai, "Hello, John!")
    assert str(mt) == f"('{Role.AI.cadenai}', 'Hello, {{name}}!')"

def test_message_template_formatting_openai():
    mt = MessageTemplate(role=Role.AI, content="Hello, {name}!")
    assert mt.format(name="John",openai_format=True) == {"role" : Role.AI.openai, "content" : "Hello, John!"}


def test_chatprompt_template_formatting():
    messages = [
        (Role.SYSTEM.cadenai, "You are a helpful AI bot. Your name is {name}."),
        (Role.HUMAN.cadenai, "Hello, how are you doing?"),
        (Role.AI.cadenai, "I'm doing well, thanks!"),
        (Role.HUMAN.cadenai, "{user_input}")
    ]
    cpt = ChatPromptTemplate.from_messages(input_variables=["name", "user_input"], messages=messages)

    formatted = cpt.format(name="Bot", user_input="Tell me more about AI")
    expected = [
        (Role.SYSTEM.cadenai, "You are a helpful AI bot. Your name is Bot."),
        (Role.HUMAN.cadenai, "Hello, how are you doing?"),
        (Role.AI.cadenai, "I'm doing well, thanks!"),
        (Role.HUMAN.cadenai, "Tell me more about AI")
    ]
    assert formatted == expected
    assert str(cpt) == f"[('{Role.SYSTEM.cadenai}', 'You are a helpful AI bot. Your name is {{name}}.'), ('{Role.HUMAN.cadenai}', 'Hello, how are you doing?'), ('{Role.AI.cadenai}', \"I'm doing well, thanks!\"), ('{Role.HUMAN.cadenai}', '{{user_input}}')]"

def test_chatprompt_template_formatting_openai_input():
    messages = [
        (Role.SYSTEM.openai, "You are a helpful AI bot. Your name is {name}."),
        (Role.HUMAN.openai, "Hello, how are you doing?"),
        (Role.AI.openai, "I'm doing well, thanks!"),
        (Role.HUMAN.openai, "{user_input}")
    ]
    cpt = ChatPromptTemplate.from_messages(input_variables=["name", "user_input"], messages=messages)

    formatted = cpt.format(name="Bot", user_input="Tell me more about AI")
    expected = [
        (Role.SYSTEM.cadenai, "You are a helpful AI bot. Your name is Bot."),
        (Role.HUMAN.cadenai, "Hello, how are you doing?"),
        (Role.AI.cadenai, "I'm doing well, thanks!"),
        (Role.HUMAN.cadenai, "Tell me more about AI")
    ]
    assert formatted == expected

def test_chatprompt_template_formatting_openai_output():
    messages = [
        (Role.SYSTEM.cadenai, "You are a helpful AI bot. Your name is {name}."),
        (Role.HUMAN.cadenai, "Hello, how are you doing?"),
        (Role.AI.cadenai, "I'm doing well, thanks!"),
        (Role.HUMAN.cadenai, "{user_input}")
    ]
    cpt = ChatPromptTemplate.from_messages(input_variables=["name", "user_input"], messages=messages)

    formatted = cpt.format(name="Bot", user_input="Tell me more about AI",openai_format=True)
    expected = [
        {"role" : Role.SYSTEM.openai, "content" : "You are a helpful AI bot. Your name is Bot."},
        {"role" : Role.HUMAN.openai, "content" : "Hello, how are you doing?"},
        {"role" : Role.AI.openai, "content" : "I'm doing well, thanks!"},
        {"role" : Role.HUMAN.openai, "content" : "Tell me more about AI"}
    ]
    assert formatted == expected


def test_chatprompt_template_add_system_message():

    cpt = ChatPromptTemplate(input_variables=[], messages_template=[])
    cpt.add_system_message("This is a system message.")

    assert len(cpt.messages_template) == 1
    assert cpt.messages_template[0].role == Role.SYSTEM
    assert cpt.messages_template[0].content == "This is a system message."
    assert cpt.input_variables == []

def test_chatprompt_template_add_system_message_with_input_variables():

    cpt = ChatPromptTemplate(input_variables=[], messages_template=[])

    cpt.add_system_message("Hello, {name}.", input_variables=["name"])

    assert len(cpt.messages_template) == 1
    assert cpt.messages_template[0].role == Role.SYSTEM
    assert cpt.messages_template[0].content == "Hello, {name}."
    assert "name" in cpt.input_variables

def test_chatprompt_template_add_ai_message():

    cpt = ChatPromptTemplate(input_variables=[], messages_template=[])
    cpt.add_ai_message("This is an ai message.")

    assert len(cpt.messages_template) == 1
    assert cpt.messages_template[0].role == Role.AI
    assert cpt.messages_template[0].content == "This is an ai message."
    assert cpt.input_variables == []

def test_chatprompt_template_add_ai_message_with_input_variables():

    cpt = ChatPromptTemplate(input_variables=[], messages_template=[])

    cpt.add_ai_message("Hello, {name}.", input_variables=["name"])

    assert len(cpt.messages_template) == 1
    assert cpt.messages_template[0].role == Role.AI
    assert cpt.messages_template[0].content == "Hello, {name}."
    assert "name" in cpt.input_variables

def test_chatprompt_template_add_human_message():

    cpt = ChatPromptTemplate(input_variables=[], messages_template=[])
    cpt.add_human_message("This is a human message.")

    assert len(cpt.messages_template) == 1
    assert cpt.messages_template[0].role == Role.HUMAN
    assert cpt.messages_template[0].content == "This is a human message."
    assert cpt.input_variables == []

def test_chatprompt_template_add_human_message_with_input_variables():

    cpt = ChatPromptTemplate(input_variables=[], messages_template=[])

    cpt.add_human_message("Hello, {name}.", input_variables=["name"])

    assert len(cpt.messages_template) == 1
    assert cpt.messages_template[0].role == Role.HUMAN
    assert cpt.messages_template[0].content == "Hello, {name}."
    assert "name" in cpt.input_variables