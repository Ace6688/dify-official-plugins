from pathlib import Path

import yaml
from anthropic.types import Message, TextBlock, Usage
from dify_plugin.entities.model import AIModelEntity
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm import anthropic as anthropic_module
from models.llm import llm as llm_module
from models.llm.llm import AihubmixLargeLanguageModel

MODEL = "claude-opus-5"
SCHEMA_PATH = Path(__file__).parents[1] / "models" / "llm" / f"{MODEL}.yaml"


def test_claude_tool_continuation_replays_thinking_blocks_from_previous_turn(monkeypatch) -> None:
    captured: dict = {}

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Message(
                id="msg_fake",
                type="message",
                role="assistant",
                model=MODEL,
                content=[TextBlock(type="text", text="It is sunny.")],
                stop_reason="end_turn",
                stop_sequence=None,
                usage=Usage(input_tokens=10, output_tokens=5),
            )

    class _Anthropic:
        def __init__(self, **kwargs) -> None:
            self.messages = _Messages()

    monkeypatch.setattr(anthropic_module, "Anthropic", _Anthropic)

    # Simulate what the streaming handler stores at the end of turn 1 (anthropic.py:1131).
    thinking_block = {"type": "thinking", "thinking": "Need the weather tool.", "signature": "sig-1"}
    monkeypatch.setattr(llm_module.anthropic_llm, "previous_thinking_blocks", [thinking_block])
    monkeypatch.setattr(llm_module.anthropic_llm, "previous_redacted_thinking_blocks", [])

    turn_2_history = [
        UserPromptMessage(content="What's the weather in Shanghai?"),
        AssistantPromptMessage(
            content="",
            tool_calls=[
                AssistantPromptMessage.ToolCall(
                    id="toolu_1",
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name="get_weather", arguments='{"city": "Shanghai"}'
                    ),
                )
            ],
        ),
        ToolPromptMessage(content="sunny", tool_call_id="toolu_1"),
    ]

    schemas = [AIModelEntity.model_validate(yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8")))]
    llm = AihubmixLargeLanguageModel(schemas)
    with llm.timing_context():
        llm._invoke(
            model=MODEL,
            credentials={"api_key": "fake-key-for-test", "api_url": "https://example.invalid"},
            prompt_messages=turn_2_history,
            model_parameters={"max_tokens": 64, "thinking": True, "effort": "high"},
            stream=False,
        )

    assistant_turn = captured["messages"][1]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"][0] == thinking_block
    assert assistant_turn["content"][1]["type"] == "tool_use"
    assert assistant_turn["content"][1]["id"] == "toolu_1"
