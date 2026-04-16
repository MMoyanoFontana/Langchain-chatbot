import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";

export async function GET(request: NextRequest) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  const q = request.nextUrl.searchParams.get("q") ?? "";

  try {
    const upstream = await backendFetchFromRoute(
      `/users/me/messages/search?q=${encodeURIComponent(q)}`,
      { method: "GET", headers: { Accept: "application/json" } },
      sessionToken
    );

    if (!upstream.ok) {
      return NextResponse.json(
        { error: await parseUpstreamError(upstream, "Search failed.") },
        { status: upstream.status }
      );
    }

    return NextResponse.json(await upstream.json(), {
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return NextResponse.json({ error: "Unable to reach backend." }, { status: 502 });
  }
}
