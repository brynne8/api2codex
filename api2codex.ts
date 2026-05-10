/**
 * api2codex - OpenAI Responses API to Chat Completions Proxy
 *
 * A lightweight proxy that converts OpenAI Responses API (/v1/responses)
 * requests into Chat Completions (/v1/chat/completions) and converts
 * the responses back. Supports both streaming and non-streaming modes,
 * function calling, reasoning content, and multi-turn tool conversations.
 *
 * Run: bun run api2codex.ts
 */

const VERSION = "0.1.0";

// ── Configuration ──────────────────────────────────────────────────────────

const UPSTREAM_BASE_URL = process.env.UPSTREAM_BASE_URL ?? "";
const UPSTREAM_API_KEY  = process.env.UPSTREAM_API_KEY  ?? "";
const DEFAULT_MODEL     = process.env.DEFAULT_MODEL     ?? "";
const HOST              = process.env.HOST              ?? "0.0.0.0";
const PORT              = parseInt(process.env.PORT     ?? "8000", 10);
const DEBUG             = ["1","true","yes"].includes((process.env.DEBUG ?? "").toLowerCase());

// ── Logging ────────────────────────────────────────────────────────────────

function ts() { return new Date().toISOString(); }
const log = {
  info:  (...a: unknown[]) => console.log( `${ts()} [INFO] `, ...a),
  error: (...a: unknown[]) => console.error(`${ts()} [ERROR]`, ...a),
  debug: (...a: unknown[]) => { if (DEBUG) console.log(`${ts()} [DEBUG]`, ...a); },
};

// ── Helpers ────────────────────────────────────────────────────────────────

function makeId(prefix = "resp"): string {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
}

function sse(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

// ── Convert Responses API input → Chat Completions messages ───────────────

interface Message {
  role: string;
  content: string | null;
  tool_calls?: unknown[];
  tool_call_id?: string;
}

function inputToMessages(body: Record<string, unknown>): Message[] {
  const instructions = body.instructions as string | undefined;
  const inp = body.input ?? "";
  const messages: Message[] = [];

  if (instructions) messages.push({ role: "system", content: instructions });

  if (typeof inp === "string") {
    messages.push({ role: "user", content: inp });
    return messages;
  }

  if (!Array.isArray(inp)) return messages;

  const pendingToolCalls: unknown[] = [];

  function flushToolCalls() {
    if (pendingToolCalls.length) {
      messages.push({ role: "assistant", content: null, tool_calls: [...pendingToolCalls] });
      pendingToolCalls.length = 0;
    }
  }

  for (const item of inp) {
    if (typeof item === "string") {
      flushToolCalls();
      messages.push({ role: "user", content: item });
      continue;
    }
    if (typeof item !== "object" || item === null) continue;

    const it = item as Record<string, unknown>;
    const itemType = (it.type as string) ?? "";

    if (itemType === "function_call") {
      pendingToolCalls.push({
        id:       it.call_id ?? it.id ?? "",
        type:     "function",
        function: { name: it.name ?? "", arguments: it.arguments ?? "{}" },
      });
      continue;
    }

    if (itemType === "function_call_output") {
      flushToolCalls();
      messages.push({ role: "tool", tool_call_id: it.call_id as string ?? "", content: it.output as string ?? "" });
      continue;
    }

    flushToolCalls();
    let role = (it.role as string) ?? "user";
    if (role === "developer") role = "system";

    let content = it.content ?? "";
    if (Array.isArray(content)) {
      content = (content as Record<string, unknown>[]).map(c => {
        const ct = (c.type as string) ?? "";
        if (ct === "input_text")  return (c.text as string) ?? "";
        if (ct === "input_image") return "[image]";
        return (c.text as string) ?? JSON.stringify(c);
      }).join("\n");
    }

    messages.push({ role, content: content as string });
  }

  flushToolCalls();
  return messages;
}

interface ChatRequest {
  model: string;
  messages: Message[];
  stream: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  tools?: unknown[];
}

function buildChatRequest(body: Record<string, unknown>): ChatRequest {
  const req: ChatRequest = {
    model:    (body.model as string) || DEFAULT_MODEL,
    messages: inputToMessages(body),
    stream:   (body.stream as boolean) ?? false,
  };

  if (body.temperature != null)      req.temperature = body.temperature as number;
  if (body.max_output_tokens != null) req.max_tokens  = body.max_output_tokens as number;
  if (body.top_p != null)            req.top_p        = body.top_p as number;

  const rawTools = body.tools as Record<string, unknown>[] | undefined;
  if (rawTools?.length) {
    const tools: unknown[] = [];
    for (const t of rawTools) {
      if (t.type !== "function") continue;
      const fn = (t.function as Record<string, unknown>) ?? t;
      tools.push({
        type: "function",
        function: {
          name:        (fn.name as string)        ?? "",
          description: (fn.description as string) ?? "",
          parameters:  fn.parameters              ?? {},
        },
      });
    }
    if (tools.length) req.tools = tools;
  }

  return req;
}

// ── Convert Chat Completions response → Responses API response ─────────────

function chatResponseToResponses(chatResp: Record<string, unknown>, model: string, respId: string) {
  const choice  = ((chatResp.choices as unknown[])?.[0] ?? {}) as Record<string, unknown>;
  const message = (choice.message as Record<string, unknown>) ?? {};
  const contentText = (message.content as string) ?? "";

  const outputItem: Record<string, unknown> = {
    id:      makeId("msg"),
    type:    "message",
    role:    "assistant",
    status:  "completed",
    content: [{ type: "output_text", text: contentText, annotations: [] }],
  };

  const toolCalls = message.tool_calls as Record<string, unknown>[] | undefined;
  if (toolCalls?.length) {
    for (const tc of toolCalls) {
      const fn = (tc.function as Record<string, unknown>) ?? {};
      const id = (tc.id as string) ?? makeId("call");
      (outputItem.content as unknown[]).push({
        type: "function_call", id, call_id: id,
        name: (fn.name as string) ?? "",
        arguments: (fn.arguments as string) ?? "{}",
      });
    }
  }

  const usage = (chatResp.usage as Record<string, number>) ?? {};
  return {
    id: respId, object: "response", created_at: Math.floor(Date.now() / 1000),
    status: "completed", model,
    output: [outputItem],
    usage: {
      input_tokens:  usage.prompt_tokens     ?? 0,
      output_tokens: usage.completion_tokens ?? 0,
      total_tokens:  usage.total_tokens      ?? 0,
    },
    parallel_tool_calls: true,
    previous_response_id: null,
    reasoning: { effort: "medium", summary: "auto" },
    text: { format: { type: "text" } },
    tools: [], truncation: "disabled",
  };
}

// ── Streaming: Chat Completions SSE → Responses API SSE ───────────────────

async function* streamChatToResponses(chatReq: ChatRequest, model: string, respId: string): AsyncGenerator<string> {
  const created = Math.floor(Date.now() / 1000);
  const msgId   = makeId("msg");

  let fullText  = "";
  let totalIn   = 0, totalOut = 0;
  let outputIdx = 0;
  let msgClosed = false;

  const activeToolCalls  = new Map<number, { id: string; name: string; arguments: string }>();
  const completedToolCalls: { id: string; name: string; arguments: string }[] = [];

  function* closeMsgItem(): Generator<string> {
    if (msgClosed) return;
    msgClosed = true;
    yield sse({ type: "response.content_part.done", output_index: 0, content_index: 0,
      part: { type: "output_text", text: fullText, annotations: [] } });
    yield sse({ type: "response.output_item.done", output_index: 0,
      item: { id: msgId, type: "message", role: "assistant", status: "completed",
        content: [{ type: "output_text", text: fullText, annotations: [] }] } });
    outputIdx = 1;
  }

  const emptyResp = { id: respId, object: "response", created_at: created,
    status: "in_progress", model, output: [], usage: null };
  yield sse({ type: "response.created",     response: emptyResp });
  yield sse({ type: "response.in_progress", response: emptyResp });
  yield sse({ type: "response.output_item.added", output_index: 0,
    item: { id: msgId, type: "message", role: "assistant", status: "in_progress", content: [] } });
  yield sse({ type: "response.content_part.added", output_index: 0, content_index: 0,
    part: { type: "output_text", text: "", annotations: [] } });

  const headers = {
    "Authorization": `Bearer ${UPSTREAM_API_KEY}`,
    "Content-Type":  "application/json",
  };

  let upstreamResp: Response;
  try {
    upstreamResp = await fetch(`${UPSTREAM_BASE_URL}/chat/completions`, {
      method: "POST", headers, body: JSON.stringify(chatReq),
    });
  } catch (err) {
    log.error("Upstream fetch failed:", err);
    yield sse({ type: "response.failed", response: { id: respId, status: "failed",
      error: { code: "server_error", message: String(err) } } });
    yield "data: [DONE]\n\n";
    return;
  }

  log.info("Upstream status:", upstreamResp.status);
  if (!upstreamResp.ok || !upstreamResp.body) {
    const errBody = await upstreamResp.text();
    log.error("Upstream error:", errBody.slice(0, 500));
    yield sse({ type: "response.failed", response: { id: respId, status: "failed",
      error: { code: "server_error", message: errBody.slice(0, 200) } } });
    yield "data: [DONE]\n\n";
    return;
  }

  const reader  = upstreamResp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  outer: while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const lines = buf.split("\n");
    buf = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const dataStr = line.slice(6).trim();
      if (dataStr === "[DONE]") {
        log.debug(`Stream DONE text_len=${fullText.length} tool_calls=${completedToolCalls.length}`);
        break outer;
      }

      let chunk: Record<string, unknown>;
      try { chunk = JSON.parse(dataStr); }
      catch { continue; }

      const choices = chunk.choices as Record<string, unknown>[] | undefined;
      if (!choices?.length) {
        const u = chunk.usage as Record<string, number> | undefined;
        if (u) { totalIn = u.prompt_tokens ?? 0; totalOut = u.completion_tokens ?? 0; }
        continue;
      }

      const delta        = (choices[0].delta as Record<string, unknown>) ?? {};
      const finishReason = choices[0].finish_reason as string | undefined;

      const reasoning = delta.reasoning_content as string | undefined;
      if (reasoning) {
        yield sse({ type: "response.reasoning_text.delta", item_id: msgId,
          output_index: 0, content_index: 0, delta: reasoning });
      }

      const text = delta.content as string | undefined;
      if (text) {
        fullText += text;
        yield sse({ type: "response.output_text.delta", item_id: msgId,
          output_index: 0, content_index: 0, delta: text });
      }

      const deltaTCs = delta.tool_calls as Record<string, unknown>[] | undefined;
      if (deltaTCs?.length) {
        for (const tc of deltaTCs) {
          const tcIdx = (tc.index as number) ?? 0;
          const tcId  = tc.id as string | undefined;
          const fn    = (tc.function as Record<string, unknown>) ?? {};

          const existingIds = [...activeToolCalls.values()].map(v => v.id);
          if (tcId && !existingIds.includes(tcId)) {
            for (const ev of closeMsgItem()) yield ev;
            activeToolCalls.set(tcIdx, { id: tcId, name: (fn.name as string) ?? "", arguments: (fn.arguments as string) ?? "" });
            log.debug(`Tool call started: ${fn.name} id=${tcId}`);
            yield sse({ type: "response.output_item.added", output_index: outputIdx + tcIdx,
              item: { id: tcId, type: "function_call", call_id: tcId,
                name: fn.name ?? "", arguments: "", status: "in_progress" } });
          } else if (activeToolCalls.has(tcIdx)) {
            const argsDelta = (fn.arguments as string) ?? "";
            activeToolCalls.get(tcIdx)!.arguments += argsDelta;
            yield sse({ type: "response.function_call_arguments.delta",
              item_id: activeToolCalls.get(tcIdx)!.id,
              output_index: outputIdx + tcIdx, delta: argsDelta });
          }
        }
      }

      if (finishReason === "tool_calls") {
        for (const ev of closeMsgItem()) yield ev;
        for (const [tcIdx, tcInfo] of [...activeToolCalls.entries()].sort(([a],[b]) => a - b)) {
          completedToolCalls.push(tcInfo);
          yield sse({ type: "response.function_call_arguments.done",
            item_id: tcInfo.id, output_index: outputIdx + tcIdx, arguments: tcInfo.arguments });
          yield sse({ type: "response.output_item.done", output_index: outputIdx + tcIdx,
            item: { id: tcInfo.id, type: "function_call", call_id: tcInfo.id,
              name: tcInfo.name, arguments: tcInfo.arguments, status: "completed" } });
        }
        activeToolCalls.clear();
      }

      const u = chunk.usage as Record<string, number> | undefined;
      if (u) { totalIn = u.prompt_tokens ?? 0; totalOut = u.completion_tokens ?? 0; }
    }
  }

  for (const ev of closeMsgItem()) yield ev;

  for (const [tcIdx, tcInfo] of [...activeToolCalls.entries()].sort(([a],[b]) => a - b)) {
    if (!completedToolCalls.includes(tcInfo)) {
      completedToolCalls.push(tcInfo);
      yield sse({ type: "response.function_call_arguments.done",
        item_id: tcInfo.id, output_index: outputIdx + tcIdx, arguments: tcInfo.arguments });
      yield sse({ type: "response.output_item.done", output_index: outputIdx + tcIdx,
        item: { id: tcInfo.id, type: "function_call", call_id: tcInfo.id,
          name: tcInfo.name, arguments: tcInfo.arguments, status: "completed" } });
    }
  }

  const outputItems: unknown[] = [];
  if (fullText) outputItems.push({ id: msgId, type: "message", role: "assistant",
    status: "completed", content: [{ type: "output_text", text: fullText, annotations: [] }] });
  for (const tc of completedToolCalls)
    outputItems.push({ id: tc.id, type: "function_call", call_id: tc.id,
      name: tc.name, arguments: tc.arguments, status: "completed" });

  yield sse({ type: "response.completed", response: {
    id: respId, object: "response", created_at: created,
    status: "completed", model, output: outputItems,
    usage: { input_tokens: totalIn, output_tokens: totalOut, total_tokens: totalIn + totalOut },
    parallel_tool_calls: true, previous_response_id: null,
    reasoning: { effort: "medium", summary: "auto" },
    text: { format: { type: "text" } }, tools: [], truncation: "disabled",
  }});

  yield "data: [DONE]\n\n";
}

// ── Route handlers ─────────────────────────────────────────────────────────

function handleHealth(): Response {
  return Response.json({ status: "ok", version: VERSION });
}

async function handleModels(): Promise<Response> {
  const resp = await fetch(`${UPSTREAM_BASE_URL}/models`, {
    headers: { "Authorization": `Bearer ${UPSTREAM_API_KEY}` },
  });
  return new Response(resp.body, { status: resp.status, headers: { "Content-Type": "application/json" } });
}

async function handleResponses(req: Request): Promise<Response> {
  let body: Record<string, unknown>;
  try { body = await req.json(); }
  catch { return Response.json({ error: "invalid JSON" }, { status: 400 }); }

  const model  = (body.model as string) || DEFAULT_MODEL;
  const stream = (body.stream as boolean) ?? false;
  const respId = makeId("resp");

  log.info(`Request: model=${model} stream=${stream} input_type=${typeof body.input}`);

  const chatReq = buildChatRequest(body);
  log.debug(`Chat request: stream=${chatReq.stream} msgs=${chatReq.messages.length} model=${chatReq.model} tools=${chatReq.tools?.length ?? 0}`);

  if (stream) {
    const gen = streamChatToResponses(chatReq, model, respId);
    const readable = new ReadableStream<Uint8Array>({
      async pull(controller) {
        const { value, done } = await gen.next();
        if (done) { controller.close(); return; }
        controller.enqueue(new TextEncoder().encode(value));
      },
      async cancel() { await gen.return(undefined); },
    });
    return new Response(readable, {
      headers: {
        "Content-Type":  "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection":    "keep-alive",
        "X-Request-Id":  respId,
      },
    });
  }

  let upstreamResp: Response;
  try {
    upstreamResp = await fetch(`${UPSTREAM_BASE_URL}/chat/completions`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${UPSTREAM_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify(chatReq),
    });
  } catch (err) {
    return Response.json({ error: String(err) }, { status: 502 });
  }

  const chatResp: Record<string, unknown> = await upstreamResp.json();
  return Response.json(chatResponseToResponses(chatResp, model, respId));
}

// ── Main ───────────────────────────────────────────────────────────────────

if (!UPSTREAM_BASE_URL) { log.error("UPSTREAM_BASE_URL is required"); process.exit(1); }
if (!UPSTREAM_API_KEY)  { log.error("UPSTREAM_API_KEY is required");  process.exit(1); }

log.info(`Starting api2codex v${VERSION} on ${HOST}:${PORT}`);
log.info(`Upstream: ${UPSTREAM_BASE_URL}`);

Bun.serve({
  hostname: HOST,
  port:     PORT,
  routes: {
    "/health":       { GET: handleHealth },
    "/v1/models":    { GET: handleModels },
    "/v1/responses": { POST: handleResponses },
  },
  fetch() {
    return Response.json({ error: "not found" }, { status: 404 });
  },
});
