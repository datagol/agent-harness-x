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
    from example_loader import load_example

    simple_chat = load_example("01-basics/streaming_chat.py")
    monkeypatch.chdir(tmp_path)
    answers = iter(["wht 40*10", "what is 40*10", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
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
    out = capsys.readouterr().out
    assert "Error:" not in out
    assert len(requests) == 3 and all(request["stream"] for request in requests)
    result = requests[1]["messages"][2]["content"][0]
    assert result["tool_use_id"] == "math" and result["content"] == "400"
    assert not result.get("is_error")
    assert "The answer is still 400." in out


@pytest.mark.asyncio
async def test_coding_agent_streams_real_sdk_tool_call(tmp_path, monkeypatch, capsys):
    from example_loader import load_example

    coding_agent = load_example("02-tools/filesystem_tools_and_permissions.py")
    monkeypatch.chdir(tmp_path)
    messages = iter(["Write a Fibonacci program", "quit"])
    approvals = []

    def typed(prompt=""):
        # The chat prompt is exactly "You: "; anything else is the permission question.
        if prompt == "You: ":
            return next(messages)
        approvals.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", typed)
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
    out = capsys.readouterr().out
    assert "Error:" not in out
    assert len(approvals) == 1 and "write_file" in approvals[0]
    assert len(requests) == 2 and all(request["stream"] for request in requests)
    assert (tmp_path / "fibonacci.py").read_text() == PROGRAM
    assert "Created fibonacci.py." in out


# ── tool arguments stream as the model writes them ───────────────────────────

WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "streaming"])
async def test_our_tools_ask_for_their_arguments_as_they_are_written(tmp_path, streaming):
    """Without it the API holds a tool call back until it is complete, and a
    model writing a long file into write_file sends nothing for minutes: the
    stall check takes that for a dead stream and starts the reply over."""
    requests = []
    responses = [tool_message(), message([{"type": "text", "text": "Created fibonacci.py."}])]
    async with client_for(responses, requests) as client:
        async with Agent(provider=AnthropicProvider(client=client)) as agent:
            register_filesystem_tools(agent.tools, base_path=str(tmp_path))
            agent.permissions.set_permission("write_file", PermissionLevel.ALLOW)
            if streaming:
                async with agent.run_stream("Write a Fibonacci program") as stream:
                    [event async for event in stream]
                    result = await stream.result()
            else:
                result = await agent.run("Write a Fibonacci program")
    assert result.status == "completed", result.error
    for body in requests:
        assert body["tools"] and all(tool["eager_input_streaming"] is True for tool in body["tools"])


@pytest.mark.asyncio
async def test_server_tools_and_explicit_choices_are_left_alone():
    requests = []
    mine = {"name": "lookup", "description": "Look up.", "input_schema": {"type": "object"}}
    declined = {**mine, "name": "quiet", "eager_input_streaming": False}
    tools = [mine, declined, WEB_SEARCH]
    sent = deepcopy(tools)
    async with client_for([message([{"type": "text", "text": "ok"}])], requests) as client:
        provider = AnthropicProvider(client=client)
        await provider.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}],
                              system=None, tools=tools, max_tokens=1000)
    by_name = {tool["name"]: tool for tool in requests[0]["tools"]}
    assert by_name["lookup"]["eager_input_streaming"] is True
    assert by_name["quiet"]["eager_input_streaming"] is False
    assert by_name["web_search"] == WEB_SEARCH
    assert tools == sent, "the caller's tool definitions are not modified"


@pytest.mark.asyncio
async def test_it_can_be_turned_off():
    requests = []
    tools = [{"name": "lookup", "description": "Look up.", "input_schema": {"type": "object"}}]
    async with client_for([message([{"type": "text", "text": "ok"}])], requests) as client:
        provider = AnthropicProvider(client=client, eager_tool_streaming=False)
        await provider.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}],
                              system=None, tools=tools, max_tokens=1000)
    assert "eager_input_streaming" not in requests[0]["tools"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["ordinary", "streaming"])
async def test_an_endpoint_that_refuses_the_field_gets_requests_without_it(streaming):
    """Anthropic-compatible gateways may not know the field. The first refusal
    turns it off for the provider, and the same request goes again without it."""
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        if any("eager_input_streaming" in tool for tool in body.get("tools", [])):
            return httpx.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error",
                "message": "tools.0.custom.eager_input_streaming: Extra inputs are not permitted"}})
        reply = message([{"type": "text", "text": "ok"}])
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_message(reply))
        return httpx.Response(200, json=reply)

    tools = [{"name": "lookup", "description": "Look up.", "input_schema": {"type": "object"}}]
    request = dict(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "hi"}],
                   system=None, tools=tools, max_tokens=1000)
    async with AsyncAnthropic(api_key="test-key", base_url="https://anthropic.invalid", max_retries=0,
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
        provider = AnthropicProvider(client=client)
        for _ in range(2):
            if streaming:
                chunks = [chunk async for chunk in provider.stream(**request)]
                response = chunks[-1].data
            else:
                response = await provider.create(**request)
            assert response.text == "ok"
    assert len(bodies) == 3, "refused once, then never sent again"
    assert provider.eager_tool_streaming is False


# ── a corrupted streamed tool argument is answered, not fatal ────────────────


def corrupted_sse(body, garbage='{"path": "fibonacci.py", "content": @@'):
    """The SSE of ``body`` with the first tool call's arguments replaced by text
    that is not JSON at all, as a broken stream or proxy can deliver it."""
    text = sse_message(body)
    good = json.dumps(json.dumps(body["content"][1]["input"]))[1:-1]
    bad = json.dumps(garbage)[1:-1]
    assert good in text
    return text.replace(good, bad, 1)


def client_with_streams(streams, requests):
    def handle(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 8})
        requests.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=streams[len(requests) - 1])

    return AsyncAnthropic(api_key="test-key", base_url="https://anthropic.invalid", max_retries=0,
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))


@pytest.mark.asyncio
async def test_a_corrupted_tool_argument_stream_is_salvaged_by_the_provider():
    from harnessx.providers.base import INVALID_ARGUMENTS

    requests = []
    async with client_with_streams([corrupted_sse(tool_message())], requests) as client:
        provider = AnthropicProvider(client=client)
        chunks = [c async for c in provider.stream(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "go"}], system=None,
            tools=[{"name": "write_file", "description": "Write.", "input_schema": {"type": "object"}}],
            max_tokens=1000,
        )]
    response = chunks[-1].data
    assert chunks[-1].kind == "response" and str(response.stop_reason) in ("tool_use", "StopReason.TOOL_USE")
    [call] = response.tool_calls
    assert call.id == "tool_write" and call.name == "write_file"
    assert "not valid JSON" in call.input[INVALID_ARGUMENTS]
    assert response.text == "I'll write a Fibonacci program."


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["run", "run_stream"])
async def test_the_model_is_asked_to_resend_a_corrupted_tool_call(tmp_path, streaming):
    """Benchmark fault: the streamed tool arguments arrive as invalid JSON. On
    0.8.0 the run failed at once with ValueError; now the model gets an error
    result for that call and sends it again."""
    requests = []
    streams = [
        corrupted_sse(tool_message()),
        sse_message(tool_message()),
        sse_message(message([{"type": "text", "text": "Created fibonacci.py."}])),
    ]
    async with client_with_streams(streams, requests) as client:
        async with Agent(provider=AnthropicProvider(client=client)) as agent:
            register_filesystem_tools(agent.tools, base_path=str(tmp_path))
            agent.permissions.set_permission("write_file", PermissionLevel.ALLOW)
            if streaming:
                async with agent.run_stream("Write a Fibonacci program") as stream:
                    events = [event async for event in stream]
                    result = await stream.result()
                assert not any(e.type is RunEventType.ERROR for e in events)
            else:
                result = await agent.run("Write a Fibonacci program")
    assert result.status == "completed", result.error
    assert len(requests) == 3
    assert (tmp_path / "fibonacci.py").read_text() == PROGRAM
    answered = [b for b in requests[1]["messages"][-1]["content"] if b.get("type") == "tool_result"]
    assert answered and answered[0]["tool_use_id"] == "tool_write" and answered[0].get("is_error")
    assert "not valid JSON" in json.dumps(answered[0]["content"])


# ── tool calls stream as the model writes them ───────────────────────────────


@pytest.mark.asyncio
async def test_a_tool_call_streams_from_the_provider_as_it_is_written():
    from types import SimpleNamespace as NS

    from harnessx.providers.anthropic import _chunks
    from harnessx.types import ToolCallDelta

    class Events:
        def __init__(self, events):
            self.events = events

        def __aiter__(self):
            async def gen():
                for e in self.events:
                    yield e
            return gen()

    events = Events([
        NS(type="content_block_start", index=1, content_block=NS(type="tool_use", id="t1", name="search")),
        NS(type="content_block_delta", index=1, delta=NS(type="input_json_delta", partial_json='{"q":')),
        NS(type="input_json", partial_json='{"q":'),
        NS(type="content_block_delta", index=1, delta=NS(type="input_json_delta", partial_json='"x"}')),
        NS(type="content_block_stop", index=1),
        NS(type="text", text="hi"),
    ])
    chunks = [c async for c in _chunks(events)]
    assert [c.kind for c in chunks] == [
        "tool_call_delta", "tool_call_delta", "progress", "tool_call_delta", "progress", "text_delta"]
    assert chunks[0].data == ToolCallDelta("t1", "search")
    assert "".join(c.data.partial_json for c in chunks if c.kind == "tool_call_delta") == '{"q":"x"}'


@pytest.mark.asyncio
async def test_run_stream_shows_the_tool_call_before_it_is_dispatched(tmp_path):
    requests = []
    responses = [tool_message(), message([{"type": "text", "text": "Created fibonacci.py."}])]
    async with client_for(responses, requests) as client:
        async with Agent(provider=AnthropicProvider(client=client)) as agent:
            register_filesystem_tools(agent.tools, base_path=str(tmp_path))
            agent.permissions.set_permission("write_file", PermissionLevel.ALLOW)
            async with agent.run_stream("Write a Fibonacci program") as stream:
                events = [event async for event in stream]
                result = await stream.result()
    assert result.status == "completed"
    kinds = [e.type for e in events]
    first_delta = kinds.index(RunEventType.TOOL_CALL_DELTA)
    assert first_delta < kinds.index(RunEventType.TOOL_CALL_START)
    deltas = [e.data for e in events if e.type is RunEventType.TOOL_CALL_DELTA]
    assert deltas[0].name == "write_file" and deltas[0].partial_json == ""
    assert json.loads("".join(d.partial_json for d in deltas)) == {"path": "fibonacci.py", "content": PROGRAM}
