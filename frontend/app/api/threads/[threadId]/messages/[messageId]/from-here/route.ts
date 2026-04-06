import { backendFetchFromRoute, getRouteSessionToken, parseUpstreamError } from "@/lib/backend-route";

export async function DELETE(
  _req: Request,
  context: { params: Promise<{ threadId: string; messageId: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  const { threadId, messageId } = await context.params;

  const upstream = await backendFetchFromRoute(
    `/users/me/threads/${threadId}/messages/${messageId}/from-here`,
    { method: "DELETE" },
    sessionToken
  );

  if (!upstream.ok) {
    return Response.json(
      { error: await parseUpstreamError(upstream, "Failed to delete messages.") },
      { status: upstream.status }
    );
  }

  return new Response(null, { status: 204 });
}
