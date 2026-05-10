# api2codex

A lightweight proxy that translates the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) (`/v1/responses`) into [Chat Completions](https://platform.openai.com/docs/api-reference/chat) (`/v1/chat/completions`) requests, enabling tools like **OpenAI Codex CLI** to work with any Chat Completions-compatible provider (Anthropic, Mistral, local models via Ollama, etc.).

## Features

- Streaming and non-streaming responses
- Function / tool calling
- Reasoning content passthrough
- Multi-turn tool conversations
- `/v1/models` and `/health` endpoints proxied through

## Requirements

[Bun](https://bun.sh) v1.0 or later. No other dependencies.

## Usage

```bash
UPSTREAM_BASE_URL=https://api.openai.com/v1 \
UPSTREAM_API_KEY=sk-... \
bun run api2codex.ts
```

Then point your Responses API client at `http://localhost:8000`.

### With Codex CLI

```bash
OPENAI_BASE_URL=http://localhost:8000 \
OPENAI_API_KEY=placeholder \
codex ...
```

## Configuration

All configuration is via environment variables.

| Variable | Required | Default | Description |
|---|---|---|---|
| `UPSTREAM_BASE_URL` | Yes | — | Base URL of the upstream Chat Completions API, e.g. `https://api.openai.com/v1` |
| `UPSTREAM_API_KEY` | Yes | — | API key forwarded to the upstream provider |
| `DEFAULT_MODEL` | No | — | Model to use when the client doesn't specify one |
| `HOST` | No | `0.0.0.0` | Address to listen on |
| `PORT` | No | `8000` | Port to listen on |
| `DEBUG` | No | — | Set to `1`, `true`, or `yes` to enable debug logging |

## Examples

### OpenAI (passthrough)

```bash
UPSTREAM_BASE_URL=https://api.openai.com/v1 \
UPSTREAM_API_KEY=sk-... \
DEFAULT_MODEL=gpt-4o \
bun run api2codex.ts
```

### Anthropic (via their OpenAI-compatible endpoint)

```bash
UPSTREAM_BASE_URL=https://api.anthropic.com/v1 \
UPSTREAM_API_KEY=sk-ant-... \
DEFAULT_MODEL=claude-sonnet-4-5 \
bun run api2codex.ts
```

### Local model via Ollama

```bash
UPSTREAM_BASE_URL=http://localhost:11434/v1 \
UPSTREAM_API_KEY=ollama \
DEFAULT_MODEL=qwen2.5-coder:32b \
bun run api2codex.ts
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/responses` | Main proxy endpoint |
| `GET` | `/v1/models` | Proxied model list from upstream |
| `GET` | `/health` | Health check — returns `{"status":"ok","version":"..."}` |

## License

MIT