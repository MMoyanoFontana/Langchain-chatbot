import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  clearSessionCookie,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";
import type { ChatAttachmentRequest } from "@/lib/chat-attachments";

type ChatRequestBody = {
  prompt?: string;
  threadId?: string;
  title?: string;
  modelId?: string;
  providerCode?: string;
  attachments?: ChatAttachmentRequest[];
  regenerateFromMessageId?: string;
  continueFromMessageId?: string;
  compareWithUserMessageId?: string;
  systemPrompt?: string;
};

export async function POST(request: NextRequest) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  let prompt = "";
  let threadId = "";
  let title = "";
  let modelId = "";
  let providerCode = "";
  let attachments: ChatAttachmentRequest[] = [];
  let regenerateFromMessageId = "";
  let continueFromMessageId = "";
  let compareWithUserMessageId = "";
  let systemPrompt = "";

  try {
    const body = (await request.json()) as ChatRequestBody;
    prompt = body.prompt?.trim() ?? "";
    threadId = body.threadId?.trim() ?? "";
    title = body.title?.trim() ?? "";
    modelId = body.modelId?.trim() ?? "";
    providerCode = body.providerCode?.trim().toLowerCase() ?? "";
    attachments = Array.isArray(body.attachments) ? body.attachments : [];
    regenerateFromMessageId = body.regenerateFromMessageId?.trim() ?? "";
    continueFromMessageId = body.continueFromMessageId?.trim() ?? "";
    compareWithUserMessageId = body.compareWithUserMessageId?.trim() ?? "";
    systemPrompt = body.systemPrompt?.trim() ?? "";
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  if (
    !prompt &&
    attachments.length === 0 &&
    !regenerateFromMessageId &&
    !compareWithUserMessageId
  ) {
    return NextResponse.json(
      { error: "Message text or at least one attachment is required." },
      { status: 400 }
    );
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
    title?: string;
    model_id: string;
    provider_code: string;
    attachments: ChatAttachmentRequest[];
    regenerate_from_message_id?: string;
    continue_from_message_id?: string;
    compare_with_user_message_id?: string;
    system_prompt?: string;
  } = { prompt, model_id: modelId, provider_code: providerCode, attachments };
  if (threadId) {
    payload.thread_id = threadId;
  }
  if (title) {
    payload.title = title;
  }
  if (regenerateFromMessageId) {
    payload.regenerate_from_message_id = regenerateFromMessageId;
  }
  if (continueFromMessageId) {
    payload.continue_from_message_id = continueFromMessageId;
  }
  if (compareWithUserMessageId) {
    payload.compare_with_user_message_id = compareWithUserMessageId;
  }
  if (systemPrompt) {
    payload.system_prompt = systemPrompt;
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
    const upstreamUserMessageId = upstreamResponse.headers.get("x-user-message-id");
    if (upstreamUserMessageId) {
      responseHeaders.set("x-user-message-id", upstreamUserMessageId);
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
