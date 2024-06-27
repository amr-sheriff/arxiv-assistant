from __future__ import annotations

import inspect
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolMessage, ToolMessageChunk

if TYPE_CHECKING:
    from langchain_text_splitters import TextSplitter

    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.runnables.base import Runnable

AnyMessage = Union[
    AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage, ToolMessage
]


def get_buffer_string_adapted(
        messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Convert a sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
        ai_prefix: THe prefix to prepend to contents of AIMessages.

    Returns:
        A single string concatenation of all input messages.

    Example:
        .. code-block:: python

            from langchain_core import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="Hi, how are you?"),
                AIMessage(content="Good, how are you?"),
            ]
            get_buffer_string(messages)
            # -> "Human: Hi, how are you?\nAI: Good, how are you?"
    """
    string_messages = ["<s>"]
    for m in messages:
        if isinstance(m, HumanMessage) or (isinstance(m, ChatMessage) and m.role == "human") or isinstance(m, SystemMessage) or (isinstance(m, ChatMessage) and m.role == "system"):
            message = f"[INST] {m.content} [\INST]"
        elif isinstance(m, AIMessage) or (isinstance(m, ChatMessage) and m.role == "AI"):
            message = f"{m.content}"
            if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
                message += f"{m.additional_kwargs['function_call']}"
            if message != "" and message != " ":
                message += " <\s>"
        elif isinstance(m, FunctionMessage):
            message = f"Function: {m.content}"
        elif isinstance(m, ToolMessage):
            message = f"Tool: {m.content}"
        elif isinstance(m, ChatMessage):
            message = f"{m.role}: {m.content}"
        else:
            raise ValueError(f"Got unsupported message type: {m}")

        string_messages.append(message)

    return "".join(string_messages)
