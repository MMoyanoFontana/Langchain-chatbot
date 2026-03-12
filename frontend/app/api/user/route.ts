import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  clearSessionCookie,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";

type UserPatchBody = {
  email?: string;
  fullName?: string | null;
  isActive?: boolean;
};

export async function GET() {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  try {
    const upstreamResponse = await backendFetchFromRoute(
      "/users/me",
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
            "Backend user request failed."
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
      { error: "Unable to reach user backend." },
      { status: 502 }
    );
  }
}

export async function PATCH(request: NextRequest) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  let body: UserPatchBody;

  try {
    body = (await request.json()) as UserPatchBody;
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  const payload: Record<string, unknown> = {};
  if (typeof body.email === "string") {
    payload.email = body.email;
  }
  if (body.fullName === null || typeof body.fullName === "string") {
    payload.full_name = body.fullName;
  }
  if (typeof body.isActive === "boolean") {
    payload.is_active = body.isActive;
  }

  if (Object.keys(payload).length === 0) {
    return NextResponse.json({ error: "No fields to update." }, { status: 400 });
  }

  try {
    const upstreamResponse = await backendFetchFromRoute(
      "/users/me",
      {
        body: JSON.stringify(payload),
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
            "Backend user update failed."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
    }

    const responsePayload = await upstreamResponse.json();
    return NextResponse.json(responsePayload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    });
  } catch {
    return NextResponse.json(
      { error: "Unable to reach user backend." },
      { status: 502 }
    );
  }
}

export async function DELETE() {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    const response = NextResponse.json({ ok: true });
    clearSessionCookie(response);
    return response;
  }

  try {
    const upstreamResponse = await backendFetchFromRoute(
      "/users/me",
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
            "Backend user delete failed."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
    }

    const response = NextResponse.json({ ok: true });
    clearSessionCookie(response);
    return response;
  } catch {
    return NextResponse.json(
      { error: "Unable to reach user backend." },
      { status: 502 }
    );
  }
}
