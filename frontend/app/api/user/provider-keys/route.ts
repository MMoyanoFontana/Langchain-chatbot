const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"

const parseUpstreamError = async (response: Response, fallback: string) => {
  try {
    const payload = (await response.json()) as {
      detail?: string | { detail?: string }
      error?: string
    }
    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : typeof payload.detail?.detail === "string"
          ? payload.detail.detail
          : null
    return (payload.error ?? detail ?? fallback).trim()
  } catch {
    try {
      const errorText = (await response.text()).trim()
      return errorText || fallback
    } catch {
      return fallback
    }
  }
}

export async function GET() {
  const backendEndpoint = new URL("/users/dev/current/settings/api-keys", BACKEND_URL)

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: { Accept: "application/json" },
      method: "GET",
      cache: "no-store",
    })

    if (!upstreamResponse.ok) {
      return Response.json(
        { error: await parseUpstreamError(upstreamResponse, "Backend provider keys request failed.") },
        { status: upstreamResponse.status || 502 }
      )
    }

    const payload = await upstreamResponse.json()
    return Response.json(payload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    })
  } catch {
    return Response.json(
      { error: "Unable to reach provider keys backend." },
      { status: 502 }
    )
  }
}
