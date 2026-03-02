const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  const backendEndpoint = new URL("/catalog/models", BACKEND_URL);

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: { Accept: "application/json" },
      method: "GET",
      cache: "no-store",
    });

    if (!upstreamResponse.ok) {
      const errorText = await upstreamResponse.text();
      return Response.json(
        { error: errorText || "Backend models request failed." },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = await upstreamResponse.json();
    return Response.json(payload, {
      headers: {
        "Cache-Control": "no-store",
      },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    });
  } catch {
    return Response.json(
      { error: "Unable to reach models backend." },
      { status: 502 }
    );
  }
}
