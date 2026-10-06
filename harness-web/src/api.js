export async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail || data),
    );
  }
  return data;
}

export const finished = (status) =>
  ["completed", "failed", "cancelled"].includes(status);
export const initialFeed = () => ({
  seq: 0,
  log: "",
  run: null,
  text: "",
  tools: [],
  segments: [],
  files: [],
  todos: null,
  phase: null,
  phaseAt: 0,
  error: "",
  notice: "",
  usage: null,
});

export function reduceFeed(state, data) {
  if (data.seq <= state.seq) return state;
  let next = { ...state, seq: data.seq };
  if (data.type === "state") {
    next.run = data;
    if (data.pending && state.phase !== "approval") {
      next.phase = data.pending.kind === "question" ? "question" : "approval";
      next.phaseAt = Date.now();
    } else if (!data.pending && (state.phase === "approval" || state.phase === "question")) {
      next.phase = "waiting";
      next.phaseAt = Date.now();
    }
  }
  // The server announces a prompt with its own event, then repeats it inside
  // the next `state`. Acting on the announcement means the approval buttons
  // appear as soon as they are asked for, and keeps working if the follow-up
  // state event is missed: a reconnect, a trimmed buffer, or a late subscribe.
  if (data.type === "input_required") {
    const { seq, type, ...prompt } = data;
    next.run = { ...(state.run || {}), pending: prompt, status: "waiting" };
    if (state.phase !== prompt.kind && (prompt.kind === "approval" || prompt.kind === "question")) {
      next.phase = prompt.kind;
      next.phaseAt = Date.now();
    }
  }
  if (data.type === "output")
    next.log = (state.log + data.content).slice(-200000);
  if (data.type === "input_sent")
    next.log = (state.log + `\n› ${data.content}\n`).slice(-200000);
  if (data.type === "error") next.error = data.content;
  if (data.type === "gap") next.notice = data.content;
  if (data.type === "chat_files") next.files = data.files || [];
  if (data.type === "phase") {
    next.phase = data.phase;
    next.phaseAt = Date.now();
  }
  if (data.type === "chat_result") {
    next.text = data.message.content;
    next.tools = data.message.tools;
    next.segments = data.message.segments || state.segments;
    next.files = data.message.files || state.files;
    next.todos = data.message.todos || state.todos;
    next.usage = data.usage;
    next.phase = null;
    if (data.message.error) next.error = data.message.error;
  }
  if (data.type === "agent") {
    const { type, data: payload } = data.event;
    if (type === "text_delta") {
      next.text += payload;
      const last = state.segments[state.segments.length - 1];
      next.segments =
        last && last.type === "text"
          ? [...state.segments.slice(0, -1), { ...last, content: last.content + payload }]
          : [...state.segments, { type: "text", content: payload }];
      if (state.phase !== "text") {
        next.phase = "text";
        next.phaseAt = Date.now();
      }
    }
    if (type === "attempt_reset") {
      next.text = "";
      const last = state.segments[state.segments.length - 1];
      next.segments = last && last.type === "text" ? state.segments.slice(0, -1) : state.segments;
    }
    if (type === "error") next.error = payload;
    if (type === "todos_updated") next.todos = payload;
    if (type === "tool_call_start") {
      next.tools = [
        ...state.tools.filter((t) => t.id !== payload.id),
        { ...payload, status: "running" },
      ];
      next.phase = `tool:${payload.name}`;
      next.phaseAt = Date.now();
      next.segments = [...state.segments, { type: "tool", id: payload.id }];
    }
    if (type === "tool_result")
      next.tools = state.tools.map((t) =>
        t.id === payload.tool_call_id
          ? {
              ...t,
              content: payload.content,
              status: payload.is_error ? "failed" : "completed",
            }
          : t,
      );
    if (type === "run_result") {
      if (payload.output) next.text = payload.output;
      next.usage = payload.usage;
      next.phase = null;
    }
  }
  return next;
}
