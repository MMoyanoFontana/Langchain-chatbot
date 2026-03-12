import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";

type UpdateThreadBody = {
  title?: string;
};

export async function GET(
  _request: Request,
  context: { params: Promise<{ threadId: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { threadId } = await context.params;

  try {
    const upstreamResponse = await backendFetchFromRoute(
      `/users/me/threads/${threadId}`,
      {
        headers: { Accept: "application/json" },
        method: "GET",
      },
      sessionToken
    );

    if (!upstreamResponse.ok) {
      return NextResponse.json(
        {
          error: await parseUpstreamError(
            upstreamResponse,
            "Backend thread request failed."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = await upstreamResponse.json();
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    });
  } catch {
    return NextResponse.json(
      { error: "Unable to reach threads backend." },
      { status: 502 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ threadId: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { threadId } = await context.params;
  let body: UpdateThreadBody;

  try {
    body = (await request.json()) as UpdateThreadBody;
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  const title = body.title?.trim() ?? "";
  if (!title) {
    return NextResponse.json({ error: "Title is required." }, { status: 400 });
  }

  try {
    const upstreamResponse = await backendFetchFromRoute(
      `/users/me/threads/${threadId}`,
      {
        body: JSON.stringify({ title }),
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
            "Backend thread update failed."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = await upstreamResponse.json();
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    });
  } catch {
    return NextResponse.json(
      { error: "Unable to reach threads backend." },
      { status: 502 }
    );
  }
}

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ threadId: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { threadId } = await context.params;

  try {
    const upstreamResponse = await backendFetchFromRoute(
      `/users/me/threads/${threadId}`,
      {
        headers: { Accept: "application/json" },
        method: "DELETE",
      },
      sessionToken
    );

    if (!upstreamResponse.ok) {
      return NextResponse.json(
        {
          error: await parseUpstreamError(
            upstreamResponse,
            "Backend thread delete failed."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
    }

    return new Response(null, { status: 204 });
  } catch {
    return NextResponse.json(
      { error: "Unable to reach threads backend." },
      { status: 502 }
    );
  }
}
