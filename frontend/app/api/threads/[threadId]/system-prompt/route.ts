import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";

type UpdateSystemPromptBody = {
  system_prompt?: string | null;
};

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ threadId: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { threadId } = await context.params;
  let body: UpdateSystemPromptBody;

  try {
    body = (await request.json()) as UpdateSystemPromptBody;
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  const systemPrompt =
    typeof body.system_prompt === "string" ? body.system_prompt.trim() || null : null;

  try {
    const upstreamResponse = await backendFetchFromRoute(
      `/users/me/threads/${threadId}/system-prompt`,
      {
        body: JSON.stringify({ system_prompt: systemPrompt }),
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        method: "PATCH",
      },
      sessionToken
    );

    if (!upstreamResponse.ok) {
      return NextResponse.json(
        {
          error: await parseUpstreamError(
            upstreamResponse,
            "Failed to update system prompt."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = await upstreamResponse.json();
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
    });
  } catch {
    return NextResponse.json(
      { error: "Unable to reach threads backend." },
      { status: 502 }
    );
  }
}
