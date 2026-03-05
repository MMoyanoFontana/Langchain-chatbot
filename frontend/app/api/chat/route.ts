import { NextRequest } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

type ChatRequestBody = {
  prompt?: string;
  threadId?: string;
  modelId?: string;
  providerCode?: string;
};

const parseUpstreamError = async (response: Response, fallback: string) => {
  try {
    const payload = (await response.json()) as {
      detail?: string | { detail?: string };
      error?: string;
    };
    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : typeof payload.detail?.detail === "string"
          ? payload.detail.detail
          : null;
    return (payload.error ?? detail ?? fallback).trim();
  } catch {
    try {
      const errorText = (await response.text()).trim();
      return errorText || fallback;
    } catch {
      return fallback;
    }
  }
};

export async function POST(request: NextRequest) {
  let prompt = "";
  let threadId = "";
  let modelId = "";
  let providerCode = "";

  try {
    const body = (await request.json()) as ChatRequestBody;
    prompt = body.prompt?.trim() ?? "";
    threadId = body.threadId?.trim() ?? "";
    modelId = body.modelId?.trim() ?? "";
    providerCode = body.providerCode?.trim().toLowerCase() ?? "";
  } catch {
    return Response.json({ error: "Invalid request body." }, { status: 400 });
  }

  if (!prompt) {
    return Response.json({ error: "Prompt is required." }, { status: 400 });
  }
  if (!modelId) {
    return Response.json({ error: "Model is required." }, { status: 400 });
  }
  if (!providerCode) {
    return Response.json({ error: "Provider code is required." }, { status: 400 });
  }

  const backendEndpoint = new URL("/chat", BACKEND_URL);
  const payload: {
    prompt: string;
    thread_id?: string;
    model_id: string;
    provider_code: string;
  } = { prompt, model_id: modelId, provider_code: providerCode };
  if (threadId) {
    payload.thread_id = threadId;
  }

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: {
        Accept: "text/plain",
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify(payload),
    });

    if (!upstreamResponse.ok || !upstreamResponse.body) {
      return Response.json(
        { error: await parseUpstreamError(upstreamResponse, "Backend chat request failed.") },
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
