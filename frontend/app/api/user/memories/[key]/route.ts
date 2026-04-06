import { backendFetchFromRoute, getRouteSessionToken, parseUpstreamError } from "@/lib/backend-route";

export async function PATCH(
  req: Request,
  context: { params: Promise<{ key: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  const { key } = await context.params;
  const body = await req.json();

  const upstream = await backendFetchFromRoute(
    `/users/me/memories/${encodeURIComponent(key)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    sessionToken
  );

  if (!upstream.ok) {
    return Response.json(
      { error: await parseUpstreamError(upstream, "Failed to update memory.") },
      { status: upstream.status }
    );
  }

  return Response.json(await upstream.json());
}

export async function DELETE(
  _req: Request,
  context: { params: Promise<{ key: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  const { key } = await context.params;

  const upstream = await backendFetchFromRoute(
    `/users/me/memories/${encodeURIComponent(key)}`,
    { method: "DELETE" },
    sessionToken
  );

  if (!upstream.ok) {
    return Response.json(
      { error: await parseUpstreamError(upstream, "Failed to delete memory.") },
      { status: upstream.status }
    );
  }

  return new Response(null, { status: 204 });
}
