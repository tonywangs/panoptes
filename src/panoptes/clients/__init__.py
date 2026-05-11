"""Provider-agnostic async LLM clients for PANOPTES.

All clients implement the `LLMClient` Protocol in `base.py`. The pipeline only
talks to that Protocol — adding a new provider means writing one concrete
class and registering it; no other code changes.
"""

from panoptes.clients.base import (
    CompletionResponse,
    LLMClient,
    Message,
    SystemBlock,
    ToolChoice,
    ToolSpec,
    price_call,
)

__all__ = [
    "CompletionResponse",
    "LLMClient",
    "Message",
    "SystemBlock",
    "ToolChoice",
    "ToolSpec",
    "price_call",
]
