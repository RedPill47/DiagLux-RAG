"""DiagLux answering harness: prompts, LLM clients, letter parsing, runners.

See docs/CONTRACTS.md for the prompt template and the preds_{config_id}.jsonl schema.
"""

from diaglux.answering.prompts import (
    PROMPT_TEMPLATE,
    CLOSED_BOOK_TEMPLATE,
    build_prompt,
    prompt_template_hash,
)
from diaglux.answering.parsing import parse_letter
from diaglux.answering.clients import (
    LLMClient,
    OpenAICompatClient,
    AnthropicClient,
    MockClient,
)
from diaglux.answering.runner import RunConfig, compute_config_id, run

__all__ = [
    "PROMPT_TEMPLATE",
    "CLOSED_BOOK_TEMPLATE",
    "build_prompt",
    "prompt_template_hash",
    "parse_letter",
    "LLMClient",
    "OpenAICompatClient",
    "AnthropicClient",
    "MockClient",
    "RunConfig",
    "compute_config_id",
    "run",
]
