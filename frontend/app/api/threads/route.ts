import { NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";

export async function GET() {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  try {
    const upstreamResponse = await backendFetchFromRoute(
      "/users/me/threads",
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
            "Backend threads request failed."
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
