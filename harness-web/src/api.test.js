import test from "node:test";
import assert from "node:assert/strict";
import { initialFeed, reduceFeed } from "./api.js";

test("replayed events do not duplicate output", () => {
  const event = { seq: 1, type: "output", content: "hello" };
  const feed = reduceFeed(initialFeed(), event);
  assert.equal(reduceFeed(feed, event).log, "hello");
});

test("text and tool calls keep the order they happened in", () => {
  const agent = (seq, type, data) => ({ seq, type: "agent", event: { type, data } });
  let feed = initialFeed();
  for (const item of [
    agent(1, "text_delta", "Searching"),
    agent(2, "text_delta", " now."),
    agent(3, "tool_call_start", { id: "s", name: "search" }),
    agent(4, "tool_result", { tool_call_id: "s", content: "hits" }),
    agent(5, "text_delta", "Found it."),
    agent(6, "attempt_reset", null),
    agent(7, "text_delta", "Found it again."),
  ])
    feed = reduceFeed(feed, item);
  assert.deepEqual(
    feed.segments.map((s) => (s.type === "text" ? s.content : `tool:${s.id}`)),
    ["Searching now.", "tool:s", "Found it again."],
  );
});

test("the feed tracks what the turn is doing between visible events", () => {
  const agent = (seq, type, data) => ({ seq, type: "agent", event: { type, data } });
  let feed = reduceFeed(initialFeed(), { seq: 1, type: "phase", phase: "model" });
  assert.equal(feed.phase, "model");
  feed = reduceFeed(feed, agent(2, "text_delta", "Now I"));
  assert.equal(feed.phase, "text");
  feed = reduceFeed(feed, agent(3, "tool_call_start", { id: "t", name: "generate_file" }));
  assert.equal(feed.phase, "tool:generate_file");
  feed = reduceFeed(feed, { seq: 4, type: "phase", phase: "waiting" });
  assert.equal(feed.phase, "waiting");
  feed = reduceFeed(feed, { seq: 5, type: "chat_result", message: { content: "Done", tools: [] }, usage: {} });
  assert.equal(feed.phase, null);
});

test("files attached during a turn survive into the final message", () => {
  let feed = reduceFeed(initialFeed(), {
    seq: 1,
    type: "chat_files",
    files: [{ name: "report.html", url: "/api/chats/c/downloads/ab12/report.html", kind: "download" }],
  });
  assert.equal(feed.files.length, 1);
  feed = reduceFeed(feed, {
    seq: 2,
    type: "chat_result",
    message: { content: "Done", tools: [], files: feed.files },
    usage: {},
  });
  assert.equal(feed.files[0].url, "/api/chats/c/downloads/ab12/report.html");
});

test("tool results correlate by call ID even when they arrive out of order", () => {
  const event = (seq, type, data) => ({
    seq,
    type: "agent",
    event: { type, data },
  });
  let feed = initialFeed();
  for (const item of [
    event(1, "tool_call_start", { id: "a", name: "first" }),
    event(2, "tool_call_start", { id: "b", name: "second" }),
    event(3, "tool_result", { tool_call_id: "b", content: "second result" }),
    event(4, "tool_result", {
      tool_call_id: "a",
      content: "first error",
      is_error: true,
    }),
  ])
    feed = reduceFeed(feed, item);
  assert.deepEqual(
    feed.tools.map((tool) => [tool.id, tool.status, tool.content]),
    [
      ["a", "failed", "first error"],
      ["b", "completed", "second result"],
    ],
  );
});

test("model retry clears partial text; completed result is authoritative", () => {
  let feed = reduceFeed(initialFeed(), {
    seq: 1,
    type: "agent",
    event: { type: "text_delta", data: "partial" },
  });
  feed = reduceFeed(feed, {
    seq: 2,
    type: "agent",
    event: { type: "attempt_reset", data: {} },
  });
  assert.equal(feed.text, "");
  feed = reduceFeed(feed, {
    seq: 3,
    type: "chat_result",
    message: { content: "Final answer", tools: [] },
    usage: { output_tokens: 3 },
  });
  assert.equal(feed.text, "Final answer");
  assert.equal(feed.usage.output_tokens, 3);
});

test("completed state clears a waiting approval", () => {
  let feed = reduceFeed(initialFeed(), {
    seq: 1,
    type: "state",
    status: "waiting",
    pending: { id: "approval" },
  });
  feed = reduceFeed(feed, {
    seq: 2,
    type: "state",
    status: "completed",
    pending: null,
  });
  assert.equal(feed.run.pending, null);
});

test("output retention is bounded and a replay gap is visible", () => {
  let feed = reduceFeed(initialFeed(), {
    seq: 1,
    type: "output",
    content: "x".repeat(210000),
  });
  assert.equal(feed.log.length, 200000);
  feed = reduceFeed(feed, {
    seq: 3,
    type: "gap",
    content: "Earlier output truncated",
  });
  assert.equal(feed.notice, "Earlier output truncated");
});
