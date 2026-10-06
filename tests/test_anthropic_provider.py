"""Exercise real Anthropic SDK parsing with in-memory HTTP responses."""

import json
import socket
from copy import deepcopy

import anthropic

# anthropic >= 1.0 moved its transport to the httpx2 package; an injected
# http_client must come from the same package the SDK uses.
if int(anthropic.__version__.split(".")[0]) >= 1:
    import httpx2 as httpx
else:
    import httpx
import pytest
from anthropic import AsyncAnthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import (
    ParsedTextBlock,
    TextBlockParam,
    ThinkingBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from harnessx import Agent, PermissionLevel, ProviderResponse, RunEventType
from harnessx.builtin.filesystem import register_filesystem_tools
from harnessx.memory import ConversationMemory
from harnessx.providers.anthropic import AnthropicProvider, _from_anthropic_response


PROGRAM = "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"


def message(content, stop_reason="end_turn"):
    return {
        "id": "msg_fixture",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 8, "output_tokens": 10},
    }


def tool_message():
    return message(
        [
            {"type": "text", "text": "I'll write a Fibonacci program."},
            {
                "type": "tool_use",
                "id": "tool_write",
                "name": "write_file",
                "input": {"path": "fibonacci.py", "content": PROGRAM},
                "caller": {"type": "direct"},
            },
        ],
        stop_reason="tool_use",
    )


def sse_message(body):
    events = [
        {
            "type": "message_start",
            "message": {**body, "content": [], "stop_reason": None},
        }
    ]
    for index, block in enumerate(body["content"]):
        if block["type"] == "text":
            start = {**block, "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            assert block["type"] == "tool_use"
            start = {**block, "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
        events.extend(
            [
                {"type": "content_block_start", "index": index, "content_block": start},
                {"type": "content_block_delta", "index": index, "delta": delta},
                {"type": "content_block_stop", "index": index},
            ]
        )
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": body["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": 10},
            },
            {"type": "message_stop"},
        ]
    )
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )


def assert_request_content(messages):
    """Reject response-only fields using the installed SDK's request contracts."""
    params = {
        "text": TextBlockParam,
        "thinking": ThinkingBlockParam,
        "tool_use": ToolUseBlockParam,
        "tool_result": ToolResultBlockParam,
    }

    def check(content):
        if not isinstance(content, list):
            return
        for block in content:
            assert block.keys() <= params[block["type"]].__annotations__.keys(), block
            if block["type"] == "tool_result":
                check(block["content"])

    for entry in messages:
        check(entry["content"])


def client_for(responses, requests):
    def handle(request):
        body = json.loads(request.content)
        assert_request_content(body["messages"])
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 8})
        assert request.url.path == "/v1/messages"
        response = responses[len(requests)]
        requests.append(body)
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse_message(response),
            )
        return httpx.Response(200, json=response)

    return AsyncAnthropic(
        api_key="test-key",
        base_url="https://anthropic.invalid",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Provider regression tests must not contact live services")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("setter", [False, True], ids=["constructor", "content-setter"])
def test_sdk_content_metadata_becomes_json_without_losing_fields(setter):
    body = tool_message()
    body["content"].insert(
        0,
        {
            "type": "thinking",
            "thinking": "Use iteration.",
            "signature": "opaque-signature",
        },
    )
    body["content"][1]["citations"] = [
        {
            "type": "char_location",
            "cited_text": "Fibonacci",
            "document_index": 0,
            "document_title": "Notes",
            "start_char_index": 0,
            "end_char_index": 9,
        }
    ]
    body["content"][-1]["input"]["optional_value"] = None
    raw = AnthropicMessage(**body)
    assert type(raw.content[-1].caller).__name__ == "DirectCaller"
    assert type(raw.content[1].citations[0]).__name__ == "CitationCharLocation"

    response = _from_anthropic_response(raw)
    if setter:
        response = ProviderResponse()
        response.content = raw.content

    memory = ConversationMemory()
    memory.add_assistant_message(response.content)
    blocks = memory.get_messages()[0]["content"]
    assert blocks == [block.model_dump(mode="json") for block in raw.content]
    assert json.loads(json.dumps(blocks)) == blocks
    assert blocks[-1]["input"]["optional_value"] is None
    assert blocks[0]["signature"] == "opaque-signature"
    # Derived views remain detached from both canonical fields and the SDK response.
    blocks[-1]["caller"]["type"] = "changed"
    assert response.content[-1]["caller"] == {"type": "direct"}
    assert raw.content[-1].caller.type == "direct"


@pytest.mark.parametrize("setter", [False, True], ids=["constructor", "content-setter"])
@pytest.mark.parametrize("parsed_output", [None, {"answer": 400}])
def test_parsed_text_excludes_sdk_fields_from_canonical_content(setter, parsed_output):
    raw = ParsedTextBlock[dict[str, int]](type="text", text="400", parsed_output=parsed_output)
    response = ProviderResponse(content=[raw])
    if setter:
        response = ProviderResponse()
        response.content = [raw]
    memory = ConversationMemory()
    memory.add_assistant_message(response.content)
    blocks = memory.get_messages()[0]["content"]
    assert blocks == [{"type": "text", "text": "400", "citations": None}]
    assert raw.parsed_output == parsed_output


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "streaming"])
async def test_anthropic_tool_round_trip_and_session_snapshot(tmp_path, streaming):
    requests = []
    responses = [
        tool_message(),
        message([{"type": "text", "text": "Created fibonacci.py."}]),
    ]
    async with client_for(responses, requests) as client:
        provider = AnthropicProvider(client=client)
        async with Agent(provider=provider) as agent:
            register_filesystem_tools(agent.tools, base_path=str(tmp_path))
            agent.permissions.set_permission("write_file", PermissionLevel.ALLOW)
            if streaming:
                async with agent.run_stream("Write a Fibonacci program") as stream:
                    events = [event async for event in stream]
                    result = await stream.result()
                assert not [
                    event for event in events if event.type == RunEventType.ERROR
                ]
                assert any(event.type == RunEventType.TEXT_DELTA for event in events)
            else:
                result = await agent.run("Write a Fibonacci program")
            assert result.status == "completed", result.error
            assert result.output == "Created fibonacci.py."
            assert (tmp_path / "fibonacci.py").read_text() == PROGRAM
            assert len(requests) == 2
            # The default reply budget (32K) is past the SDK's ceiling for a
            # non-streaming call, so an ordinary run is sent as a stream too and
            # assembled into one message.
            assert requests[0]["max_tokens"] == 32_000 and requests[0].get("stream")
            history = requests[1]["messages"]
            assert history[1]["content"][-1]["caller"] == {"type": "direct"}
            assert history[2]["content"][0]["tool_use_id"] == "tool_write"
            assert not history[2]["content"][0].get("is_error")
            saved_messages = agent.memory.get_messages()
            assert_request_content(saved_messages)
            session_id = await agent.save_session(str(tmp_path / "sessions"))
        restored = await Agent.load_session(
            session_id, str(tmp_path / "sessions"), provider=provider
        )
        async with restored:
            assert restored.memory.get_messages() == saved_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "stream", "count_tokens"])
async def test_old_history_is_sanitized_without_changing_tool_data(mode):
    history = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking", "parsed_output": None},
                {
                    "type": "tool_use", "id": "tool_1", "name": "echo",
                    "input": {"type": "text", "parsed_output": {"keep": True}},
                    "caller": {"type": "direct"},
                },
            ],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result", "tool_use_id": "tool_1",
                "content": [{"type": "text", "text": "ok", "parsed_output": {"result": "ok"}}],
            }],
        },
    ]
    original = deepcopy(history)
    requests = []
    async with client_for([message([{"type": "text", "text": "done"}])], requests) as client:
        provider = AnthropicProvider(client)
        kwargs = dict(model="claude-sonnet-4-6", messages=history, system="", tools=[])
        if mode == "count_tokens":
            assert await provider.count_tokens(**kwargs) == 8
        elif mode == "stream":
            chunks = [chunk async for chunk in provider.stream(**kwargs, max_tokens=256)]
            assert chunks[-1].data.text == "done"
        else:
            assert (await provider.create(**kwargs, max_tokens=256)).text == "done"
    assert history == original
    if requests:
        assert requests[0]["messages"][1]["content"][1]["input"] == original[1]["content"][1]["input"]
        assert requests[0]["messages"][1]["content"][1]["caller"] == {"type": "direct"}


@pytest.mark.asyncio
async def test_simple_chat_calculates_with_streaming_sdk_across_turns(tmp_path, monkeypatch, capsys):
    from examples import _console, simple_chat

    monkeypatch.chdir(tmp_path)
    answers = iter(["wht 40*10", "what is 40*10", "quit"])
    monkeypatch.setattr(simple_chat, "get_user_input", lambda: next(answers))
    errors = []
    monkeypatch.setattr(simple_chat, "print_error", errors.append)
    monkeypatch.setattr(_console, "print_error", errors.append)
    requests = []
    responses = [
        message([
            {"type": "text", "text": "Let me calculate that for you!"},
            {
                "type": "tool_use", "id": "math", "name": "calculate",
                "input": {"expression": "40*10"}, "caller": {"type": "direct"},
            },
        ], stop_reason="tool_use"),
        message([{"type": "text", "text": "40 * 10 = 400"}]),
        message([{"type": "text", "text": "The answer is still 400."}]),
    ]
    async with client_for(responses, requests) as client:
        monkeypatch.setattr(
            "harnessx.core.make_provider", lambda *args, **kwargs: AnthropicProvider(client),
        )
        await simple_chat.main()
    assert not errors
    assert len(requests) == 3 and all(request["stream"] for request in requests)
    result = requests[1]["messages"][2]["content"][0]
    assert result["tool_use_id"] == "math" and result["content"] == "400"
    assert not result.get("is_error")
    assert "The answer is still 400." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_coding_agent_streams_real_sdk_tool_call(tmp_path, monkeypatch, capsys):
    from examples import _console, coding_agent

    monkeypatch.chdir(tmp_path)
    answers = iter(["Write a Fibonacci program", "quit"])
    monkeypatch.setattr(coding_agent, "get_user_input", lambda: next(answers))
    approvals = []

    def approve(prompt):
        approvals.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", approve)
    errors = []
    monkeypatch.setattr(coding_agent, "print_error", errors.append)
    monkeypatch.setattr(_console, "print_error", errors.append)
    requests = []
    responses = [
        tool_message(),
        message([{"type": "text", "text": "Created fibonacci.py."}]),
    ]
    async with client_for(responses, requests) as client:
        monkeypatch.setattr(
            "harnessx.core.make_provider",
            lambda *args, **kwargs: AnthropicProvider(client),
        )
        await coding_agent.main()
    assert not errors
    assert len(approvals) == 1 and "write_file" in approvals[0]
    assert len(requests) == 2 and all(request["stream"] for request in requests)
    assert (tmp_path / "fibonacci.py").read_text() == PROGRAM
    assert "Created fibonacci.py." in capsys.readouterr().out
