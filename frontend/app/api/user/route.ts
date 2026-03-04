import { NextRequest } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"

type UserPatchBody = {
  email?: string
  fullName?: string | null
  isActive?: boolean
}

export async function GET() {
  const backendEndpoint = new URL("/users/dev/current", BACKEND_URL)

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: { Accept: "application/json" },
      method: "GET",
      cache: "no-store",
    })

    if (!upstreamResponse.ok) {
      const errorText = await upstreamResponse.text()
      return Response.json(
        { error: errorText || "Backend user request failed." },
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
      { error: "Unable to reach user backend." },
      { status: 502 }
    )
  }
}

export async function PATCH(request: NextRequest) {
  let body: UserPatchBody

  try {
    body = (await request.json()) as UserPatchBody
  } catch {
    return Response.json({ error: "Invalid request body." }, { status: 400 })
  }

  const payload: Record<string, unknown> = {}
  if (typeof body.email === "string") {
    payload.email = body.email
  }
  if (body.fullName === null || typeof body.fullName === "string") {
    payload.full_name = body.fullName
  }
  if (typeof body.isActive === "boolean") {
    payload.is_active = body.isActive
  }

  if (Object.keys(payload).length === 0) {
    return Response.json({ error: "No fields to update." }, { status: 400 })
  }

  const backendEndpoint = new URL("/users/dev/current", BACKEND_URL)

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "PATCH",
      body: JSON.stringify(payload),
      cache: "no-store",
    })

    if (!upstreamResponse.ok) {
      const errorText = await upstreamResponse.text()
      return Response.json(
        { error: errorText || "Backend user update failed." },
        { status: upstreamResponse.status || 502 }
      )
    }

    const responsePayload = await upstreamResponse.json()
    return Response.json(responsePayload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    })
  } catch {
    return Response.json(
      { error: "Unable to reach user backend." },
      { status: 502 }
    )
  }
}

export async function DELETE() {
  const backendEndpoint = new URL("/users/dev/current", BACKEND_URL)

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: { Accept: "application/json" },
      method: "DELETE",
      cache: "no-store",
    })

    if (!upstreamResponse.ok) {
      const errorText = await upstreamResponse.text()
      return Response.json(
        { error: errorText || "Backend user delete failed." },
        { status: upstreamResponse.status || 502 }
      )
    }

    return new Response(null, { status: 204 })
  } catch {
    return Response.json(
      { error: "Unable to reach user backend." },
      { status: 502 }
    )
  }
}
