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
  files: [],
  error: "",
  notice: "",
  usage: null,
});

export function reduceFeed(state, data) {
  if (data.seq <= state.seq) return state;
  let next = { ...state, seq: data.seq };
  if (data.type === "state") next.run = data;
  if (data.type === "output")
    next.log = (state.log + data.content).slice(-200000);
  if (data.type === "input_sent")
    next.log = (state.log + `\n› ${data.content}\n`).slice(-200000);
  if (data.type === "error") next.error = data.content;
  if (data.type === "gap") next.notice = data.content;
  if (data.type === "chat_files") next.files = data.files || [];
  if (data.type === "chat_result") {
    next.text = data.message.content;
    next.tools = data.message.tools;
    next.files = data.message.files || state.files;
    next.usage = data.usage;
    if (data.message.error) next.error = data.message.error;
  }
  if (data.type === "agent") {
    const { type, data: payload } = data.event;
    if (type === "text_delta") next.text += payload;
    if (type === "attempt_reset") next.text = "";
    if (type === "error") next.error = payload;
    if (type === "tool_call_start")
      next.tools = [
        ...state.tools.filter((t) => t.id !== payload.id),
        { ...payload, status: "running" },
      ];
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
    }
  }
  return next;
}
