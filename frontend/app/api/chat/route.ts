import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  clearSessionCookie,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";

type ChatRequestBody = {
  prompt?: string;
  threadId?: string;
  modelId?: string;
  providerCode?: string;
};

export async function POST(request: NextRequest) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

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
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  if (!prompt) {
    return NextResponse.json({ error: "Prompt is required." }, { status: 400 });
  }
  if (!modelId) {
    return NextResponse.json({ error: "Model is required." }, { status: 400 });
  }
  if (!providerCode) {
    return NextResponse.json({ error: "Provider code is required." }, { status: 400 });
  }

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
    const upstreamResponse = await backendFetchFromRoute(
      "/chat",
      {
        body: JSON.stringify(payload),
        headers: {
          Accept: "text/plain",
          "Content-Type": "application/json",
        },
        method: "POST",
      },
      sessionToken
    );

    if (!upstreamResponse.ok || !upstreamResponse.body) {
      const response = NextResponse.json(
        {
          error: await parseUpstreamError(
            upstreamResponse,
            "Backend chat request failed."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
      if (upstreamResponse.status === 401) {
        clearSessionCookie(response);
      }
      return response;
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
    return NextResponse.json(
      { error: "Unable to reach chat backend." },
      { status: 502 }
    );
  }
}
