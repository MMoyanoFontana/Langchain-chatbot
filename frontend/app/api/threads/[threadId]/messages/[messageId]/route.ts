import { backendFetchFromRoute, getRouteSessionToken, parseUpstreamError } from "@/lib/backend-route";

export async function DELETE(
  _req: Request,
  context: { params: Promise<{ threadId: string; messageId: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return Response.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { threadId, messageId } = await context.params;

  try {
    const upstream = await backendFetchFromRoute(
      `/users/me/threads/${threadId}/messages/${messageId}`,
      { method: "DELETE" },
      sessionToken
    );

    if (!upstream.ok) {
      return Response.json(
        { error: await parseUpstreamError(upstream, "Failed to delete message.") },
        { status: upstream.status }
      );
    }

    return new Response(null, { status: 204 });
  } catch {
    return Response.json({ error: "Unable to reach backend." }, { status: 502 });
  }
}
