import {
  backendFetchFromRoute,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";

export async function POST(
  _request: Request,
  context: { params: Promise<{ threadId: string; documentId: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return Response.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { threadId, documentId } = await context.params;

  try {
    const upstreamResponse = await backendFetchFromRoute(
      `/users/me/threads/${threadId}/documents/${documentId}/retry`,
      {
        headers: { Accept: "application/json" },
        method: "POST",
      },
      sessionToken
    );

    if (!upstreamResponse.ok) {
      return Response.json(
        { error: await parseUpstreamError(upstreamResponse, "Document retry failed.") },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = await upstreamResponse.json();
    return Response.json(payload, { status: upstreamResponse.status });
  } catch {
    return Response.json({ error: "Unable to reach backend." }, { status: 502 });
  }
}
