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

test("an approval wait is its own phase", () => {
  let feed = reduceFeed(initialFeed(), { seq: 1, type: "phase", phase: "waiting" });
  feed = reduceFeed(feed, { seq: 2, type: "state", status: "waiting", pending: { id: "p", kind: "approval" } });
  assert.equal(feed.phase, "approval");
  feed = reduceFeed(feed, { seq: 3, type: "state", status: "running", pending: null });
  assert.equal(feed.phase, "waiting");
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

test("an approval prompt renders from input_required alone", () => {
  // The follow-up `state` event is deliberately withheld: a reconnect or a
  // trimmed buffer can drop it, and the buttons must still appear.
  let feed = reduceFeed(initialFeed(), { seq: 1, type: "state", status: "running", pending: null });
  feed = reduceFeed(feed, {
    seq: 2, type: "input_required", id: "p1", kind: "approval",
    prompt: "Allow write_file?", tool: { name: "write_file", input: { path: "hi.txt" } },
  });
  assert.equal(feed.phase, "approval");
  assert.equal(feed.run.pending.id, "p1");
  assert.equal(feed.run.pending.kind, "approval");
  assert.equal(feed.run.pending.tool.name, "write_file");
  assert.equal(feed.run.pending.seq, undefined, "the envelope's seq is not part of the prompt");
});

test("the state event that follows an input_required does not clobber it", () => {
  let feed = reduceFeed(initialFeed(), { seq: 1, type: "input_required", id: "p1", kind: "approval", prompt: "Allow?" });
  feed = reduceFeed(feed, {
    seq: 2, type: "state", status: "waiting", id: "r1",
    pending: { id: "p1", kind: "approval", prompt: "Allow?" },
  });
  assert.equal(feed.run.pending.id, "p1");
  assert.equal(feed.run.id, "r1", "the state event still supplies the run's own fields");
});

test("the agent's plan follows its latest update and survives the final result", () => {
  const plan = (done) => ({
    todos: [
      { content: "Read the code", status: done ? "completed" : "in_progress" },
      { content: "Write the fix", status: done ? "in_progress" : "pending" },
    ],
    completed: done ? 1 : 0, total: 2, in_progress: done ? "Write the fix" : "Read the code",
  });
  let feed = reduceFeed(initialFeed(), { seq: 1, type: "agent", event: { type: "todos_updated", data: plan(false) } });
  feed = reduceFeed(feed, { seq: 2, type: "agent", event: { type: "todos_updated", data: plan(true) } });
  assert.equal(feed.todos.completed, 1);
  feed = reduceFeed(feed, { seq: 3, type: "chat_result", message: { content: "done", tools: [], segments: [] } });
  assert.equal(feed.todos.in_progress, "Write the fix", "a result without a plan keeps the one shown");
});

test("a question from the agent is its own phase, not an approval", () => {
  let feed = reduceFeed(initialFeed(), {
    seq: 1, type: "input_required", id: "q1", kind: "question",
    prompt: "Which Jev do you mean?", choices: ["Jevons paradox", "HarnessX Jev decisions"],
  });
  assert.equal(feed.phase, "question");
  assert.deepEqual(feed.run.pending.choices, ["Jevons paradox", "HarnessX Jev decisions"]);
  feed = reduceFeed(feed, { seq: 2, type: "state", status: "running", id: "r1", pending: null });
  assert.equal(feed.phase, "waiting");
});
