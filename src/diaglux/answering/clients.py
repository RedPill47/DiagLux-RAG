"""LLM client abstractions for the answering harness.

All third-party SDK imports are lazy (inside the concrete classes) so that
the diaglux package imports and tests run without `openai`/`anthropic`
installed and without network access. Tests must use MockClient only; no
real API calls happen anywhere in the test suite.
"""

from __future__ import annotations

import abc
import os
from typing import Callable, Mapping, Optional, Sequence, Union


class LLMClient(abc.ABC):
    """Minimal interface: one prompt in, raw completion text out."""

    @abc.abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the raw model output for ``prompt``."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Provider/model info recorded in the run config sidecar."""
        return {"client": type(self).__name__}


class OpenAICompatClient(LLMClient):
    """Client for the OpenAI API or any OpenAI-compatible endpoint.

    ``base_url`` makes this usable with vLLM, Ollama, OpenRouter, etc.
    The API key is read from the environment variable named by
    ``api_key_env`` at first use, never stored in configs.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        # Adaptive parameter handling for models with stricter APIs (e.g. the
        # GPT-5 family, which requires `max_completion_tokens` and rejects any
        # `temperature` other than the default 1). These are flipped on the
        # first call if the API rejects the defaults, then cached.
        self._token_param = "max_tokens"
        self._omit_temperature = False

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # lazy import per CONTRACTS.md env notes
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "The 'openai' package is required for OpenAICompatClient. "
                    "Install it with: pip install openai"
                ) from exc
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Environment variable {self.api_key_env} is not set; "
                    "it must contain the API key."
                )
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        last_exc: Optional[Exception] = None
        # At most a few attempts: each known unsupported-parameter 400 flips one
        # setting and retries; an unrecognized error is re-raised immediately.
        for _ in range(4):
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                self._token_param: self.max_tokens,
            }
            if not self._omit_temperature:
                kwargs["temperature"] = self.temperature
            try:
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:  # narrow via message inspection below
                msg = str(exc).lower()
                adapted = False
                if (self._token_param == "max_tokens"
                        and "max_tokens" in msg and "max_completion_tokens" in msg):
                    self._token_param = "max_completion_tokens"
                    adapted = True
                if (not self._omit_temperature and "temperature" in msg
                        and ("does not support" in msg or "only the default" in msg
                             or "unsupported value" in msg)):
                    self._omit_temperature = True
                    adapted = True
                if not adapted:
                    raise
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("completion failed without an exception")  # unreachable

    def describe(self) -> dict:
        return {
            "client": "OpenAICompatClient",
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            # Effective decoding params: temperature is None when the model only
            # supports its default (recorded so the sidecar is truthful).
            "temperature": None if self._omit_temperature else self.temperature,
            "requested_temperature": self.temperature,
            "token_param": self._token_param,
            "max_tokens": self.max_tokens,
        }


class AnthropicClient(LLMClient):
    """Client for the Anthropic Messages API (lazy import of `anthropic`)."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 64,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._client = None
        # Some newer Claude models deprecate the `temperature` parameter; drop
        # it on the first call that rejects it, then cache.
        self._omit_temperature = False

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # lazy import per CONTRACTS.md env notes
            except ImportError as exc:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "The 'anthropic' package is required for AnthropicClient. "
                    "Install it with: pip install anthropic"
                ) from exc
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Environment variable {self.api_key_env} is not set; "
                    "it must contain the API key."
                )
            kwargs = {"api_key": api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._get_client()
        last_exc: Optional[Exception] = None
        for _ in range(3):
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if not self._omit_temperature:
                kwargs["temperature"] = self.temperature
            try:
                response = client.messages.create(**kwargs)
                parts = [
                    block.text
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                ]
                return "".join(parts)
            except Exception as exc:  # narrow via message inspection
                msg = str(exc).lower()
                if (not self._omit_temperature and "temperature" in msg
                        and ("deprecated" in msg or "not support" in msg
                             or "unsupported" in msg)):
                    self._omit_temperature = True
                    last_exc = exc
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("completion failed without an exception")  # unreachable

    def describe(self) -> dict:
        return {
            "client": "AnthropicClient",
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "temperature": None if self._omit_temperature else self.temperature,
            "requested_temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class MockClient(LLMClient):
    """Deterministic offline client for tests and smoke runs.

    Canned outputs, resolved in this order:

    - ``responder``: callable ``(prompt) -> str``, full control.
    - ``by_substring``: mapping; the first key found as a substring of the
      prompt selects the output (insertion order).
    - ``outputs``: sequence consumed call by call; when exhausted, falls back
      to ``default``.
    - ``default``: constant fallback (default "A").

    Every prompt is recorded in ``self.calls`` for assertions.
    """

    def __init__(
        self,
        outputs: Optional[Sequence[str]] = None,
        default: str = "A",
        by_substring: Optional[Mapping[str, str]] = None,
        responder: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._outputs = list(outputs) if outputs is not None else None
        self._index = 0
        self.default = default
        self.by_substring = dict(by_substring) if by_substring else None
        self.responder = responder
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.responder is not None:
            return self.responder(prompt)
        if self.by_substring is not None:
            for key, value in self.by_substring.items():
                if key in prompt:
                    return value
        if self._outputs is not None and self._index < len(self._outputs):
            out = self._outputs[self._index]
            self._index += 1
            return out
        return self.default

    def describe(self) -> dict:
        return {"client": "MockClient", "default": self.default}


def make_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 64,
    api_key_env: Optional[str] = None,
    mock_output: str = "A",
) -> LLMClient:
    """Factory used by the CLI. ``provider`` is openai | anthropic | mock."""
    if provider == "openai":
        return OpenAICompatClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env or "OPENAI_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "anthropic":
        return AnthropicClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env or "ANTHROPIC_API_KEY",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "mock":
        return MockClient(default=mock_output)
    raise ValueError(f"Unknown provider: {provider!r}")
