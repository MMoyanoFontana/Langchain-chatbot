import { NextRequest } from "next/server";

import { backendFetchFromRoute, getRouteSessionToken } from "@/lib/backend-route";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ threadId: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return Response.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { threadId } = await context.params;
  const format = request.nextUrl.searchParams.get("format") ?? "markdown";

  let upstream: Response;
  try {
    upstream = await backendFetchFromRoute(
      `/users/me/threads/${threadId}/export?format=${encodeURIComponent(format)}`,
      { method: "GET" },
      sessionToken
    );
  } catch {
    return Response.json({ error: "Unable to reach backend." }, { status: 502 });
  }

  if (!upstream.ok) {
    return Response.json({ error: "Export failed." }, { status: upstream.status });
  }

  const body = await upstream.arrayBuffer();
  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": upstream.headers.get("Content-Type") ?? "text/plain",
      "Content-Disposition":
        upstream.headers.get("Content-Disposition") ?? "attachment",
    },
  });
}
