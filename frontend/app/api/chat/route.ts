import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

type ChatRequestBody = {
  prompt?: string;
  threadId?: string;
};

export async function POST(request: NextRequest) {
  let prompt = "";
  let threadId = "";

  try {
    const body = (await request.json()) as ChatRequestBody;
    prompt = body.prompt?.trim() ?? "";
    threadId = body.threadId?.trim() ?? "";
  } catch {
    return Response.json({ error: "Invalid request body." }, { status: 400 });
  }

  if (!prompt) {
    return Response.json({ error: "Prompt is required." }, { status: 400 });
  }

  const backendEndpoint = new URL("/chat", BACKEND_URL);
  backendEndpoint.searchParams.set("prompt", prompt);
  if (threadId) {
    backendEndpoint.searchParams.set("thread_id", threadId);
  }

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: { Accept: "text/plain" },
      method: "GET",
    });

    if (!upstreamResponse.ok || !upstreamResponse.body) {
      const errorText = await upstreamResponse.text();
      return Response.json(
        { error: errorText || "Backend chat request failed." },
        { status: upstreamResponse.status || 502 }
      );
    }

    const responseHeaders = new Headers({
      "Cache-Control": "no-cache, no-transform",
      "Content-Type": "text/plain; charset=utf-8",
    });
    const upstreamThreadId = upstreamResponse.headers.get("x-thread-id");
    if (upstreamThreadId) {
      responseHeaders.set("x-thread-id", upstreamThreadId);
    }

    return new Response(upstreamResponse.body, {
      headers: responseHeaders,
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    });
  } catch {
    return Response.json(
      { error: "Unable to reach chat backend." },
      { status: 502 }
    );
  }
}
