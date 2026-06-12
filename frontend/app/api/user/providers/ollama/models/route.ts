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
      "/users/me/settings/providers/ollama/models",
      {
        headers: { Accept: "application/json" },
        method: "GET",
      },
      sessionToken
    );

    if (!upstreamResponse.ok) {
      return NextResponse.json(
        { error: await parseUpstreamError(upstreamResponse, "Could not fetch Ollama models.") },
        { status: upstreamResponse.status }
      );
    }

    const payload = await upstreamResponse.json();
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
    });
  } catch {
    return NextResponse.json(
      { error: "Unable to reach Ollama model server." },
      { status: 502 }
    );
  }
}
