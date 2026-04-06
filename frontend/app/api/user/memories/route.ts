import { backendFetchFromRoute, getRouteSessionToken, parseUpstreamError } from "@/lib/backend-route";

export async function GET() {
  const sessionToken = await getRouteSessionToken();

  const upstream = await backendFetchFromRoute("/users/me/memories", {}, sessionToken);

  if (!upstream.ok) {
    return Response.json(
      { error: await parseUpstreamError(upstream, "Failed to load memories.") },
      { status: upstream.status }
    );
  }

  return Response.json(await upstream.json());
}
