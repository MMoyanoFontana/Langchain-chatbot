# Frontend

Next.js 16 + React 19 + Tailwind 4. Next.js route handlers in [app/api/](app/api/) proxy every request to the FastAPI backend. See the [root README](../README.md) for architecture and product overview.

## Run

```bash
pnpm install
pnpm dev                        # :3000
pnpm lint
pnpm build
```

## Environment variables

Reads `.env.local` in this directory.

| Variable | Notes |
|---|---|
| `BACKEND_URL` | Defaults to `http://127.0.0.1:8000`. Set to your Render backend URL in prod. |

## Layout

```
app/
  (app)/chats/[chatId]/         Server component: fetches thread data, renders ChatSession
  api/                          Next.js route handlers — every route proxies to FastAPI
    chat/                         Streaming chat + attachment upload
    threads/[threadId]/           Thread CRUD, messages, documents, system prompt, export
    auth/                         Login, register, OAuth, logout
    user/                         Provider keys, memories
    messages/search/              Cross-thread message search
components/
  chat-session.tsx              Main chat UI: streaming, citations, document panel, branching
  prompt-composer.tsx           Input with file attachment + model selection
  ai-elements/                  Vercel AI SDK component library (reasoning, tool, citation, etc.)
  ui/                           shadcn/ui primitives
lib/
  backend.ts / backend-route.ts Server-side fetch helpers for proxying to FastAPI
  chat-citations.ts             Parse X-Message-Citations header
  chat-attachments.ts           Convert files to/from backend attachment format
  thread-documents.ts           Normalize backend document responses
  use-chat-models.ts            Model selection hook
```

## API route convention

Every Next.js route handler is a thin proxy to FastAPI. Use [`backendFetchFromRoute`](lib/backend-route.ts) and [`parseUpstreamError`](lib/backend-route.ts), and always `await context.params`.

```typescript
export async function DELETE(_req, context: { params: Promise<{ threadId: string }> }) {
  const sessionToken = await getRouteSessionToken();
  const { threadId } = await context.params;
  const upstream = await backendFetchFromRoute(
    `/users/me/threads/${threadId}`,
    { method: "DELETE" },
    sessionToken,
  );
  if (!upstream.ok) {
    return Response.json({ error: await parseUpstreamError(upstream, "Failed.") }, { status: upstream.status });
  }
  return new Response(null, { status: 204 });
}
```

## Streaming

The chat endpoint streams plain text from FastAPI. Citations and message IDs come back as response headers (`X-Thread-Id`, `X-User-Message-Id`, `X-Message-Citations`) which `chat-session.tsx` parses and merges into the message state.

Reasoning tokens and tool-call events are interleaved into the stream as JSON event lines and rendered inline by [reasoning.tsx](components/ai-elements/reasoning.tsx) and [tool.tsx](components/ai-elements/tool.tsx).
