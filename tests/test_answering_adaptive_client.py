"""OpenAICompatClient adapts to stricter model APIs (e.g. GPT-5 family).

Simulates the two known unsupported-parameter 400s (max_tokens ->
max_completion_tokens; temperature 0 -> default only) with a fake OpenAI
client and asserts the client flips the settings, retries, succeeds, caches
the working set, and reports the effective config in describe().
"""

import pytest

from diaglux.answering.clients import AnthropicClient, OpenAICompatClient


class _Resp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class FakeCompletions:
    """Rejects max_tokens, then temperature!=default, then succeeds; records calls."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise RuntimeError(
                "Error code: 400 - Unsupported parameter: 'max_tokens' is not "
                "supported with this model. Use 'max_completion_tokens' instead."
            )
        if kwargs.get("temperature") not in (None, 1):
            raise RuntimeError(
                "Error code: 400 - Unsupported value: 'temperature' does not "
                "support 0 with this model. Only the default (1) value is supported."
            )
        return _Resp("B")


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self):
        self.chat = FakeChat()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    c = OpenAICompatClient(model="gpt-5.5", temperature=0.0, max_tokens=64)
    c._client = FakeOpenAI()  # inject fake; skip real SDK/network
    return c


def test_adapts_token_param_and_temperature(client):
    assert client.complete("Answer:") == "B"
    calls = client._client.chat.completions.calls
    # 1: max_tokens -> 400; 2: max_completion_tokens + temp=0 -> 400; 3: ok
    assert len(calls) == 3
    assert "max_tokens" in calls[0]
    assert calls[1]["max_completion_tokens"] == 64 and calls[1]["temperature"] == 0.0
    assert "temperature" not in calls[2] and calls[2]["max_completion_tokens"] == 64


def test_caches_adapted_params_on_second_call(client):
    client.complete("Answer:")
    n_after_first = len(client._client.chat.completions.calls)
    client.complete("Answer again:")
    # second prompt needs exactly one call: the working param set is cached
    assert len(client._client.chat.completions.calls) == n_after_first + 1


def test_describe_reports_effective_config(client):
    client.complete("Answer:")
    d = client.describe()
    assert d["token_param"] == "max_completion_tokens"
    assert d["temperature"] is None          # omitted -> model default
    assert d["requested_temperature"] == 0.0  # original request preserved


def test_unrecognized_error_is_reraised(client):
    def boom(**kwargs):
        raise RuntimeError("Error code: 401 - invalid api key")

    client._client.chat.completions.create = boom
    with pytest.raises(RuntimeError, match="401"):
        client.complete("Answer:")


# --- AnthropicClient: drops a deprecated temperature parameter ---------------

class _AResp:
    def __init__(self, text):
        self.content = [type("Blk", (), {"type": "text", "text": text})()]


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs:
            raise RuntimeError(
                "Error code: 400 - {'type': 'error', 'error': {'type': "
                "'invalid_request_error', 'message': '`temperature` is "
                "deprecated for this model.'}}"
            )
        return _AResp("C")


class FakeAnthropic:
    def __init__(self):
        self.messages = FakeMessages()


@pytest.fixture
def anthropic_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    c = AnthropicClient(model="claude-opus-4-8", temperature=0.0, max_tokens=64)
    c._client = FakeAnthropic()
    return c


def test_anthropic_drops_deprecated_temperature(anthropic_client):
    assert anthropic_client.complete("Answer:") == "C"
    calls = anthropic_client._client.messages.calls
    assert len(calls) == 2  # temp -> 400, then dropped -> ok
    assert "temperature" in calls[0] and "temperature" not in calls[1]
    d = anthropic_client.describe()
    assert d["temperature"] is None and d["requested_temperature"] == 0.0


def test_anthropic_reraises_unrelated_error(anthropic_client):
    def boom(**kwargs):
        raise RuntimeError("Error code: 529 - overloaded")

    anthropic_client._client.messages.create = boom
    with pytest.raises(RuntimeError, match="529"):
        anthropic_client.complete("Answer:")
