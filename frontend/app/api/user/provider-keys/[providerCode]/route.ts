import { NextRequest } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"
const ALLOWED_PROVIDER_CODES = new Set(["openai", "anthropic", "gemini", "groq", "other"])

type UpsertProviderKeyBody = {
  apiKey?: string
  keyName?: string
  isDefault?: boolean
  isActive?: boolean
}

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ providerCode: string }> }
) {
  const { providerCode } = await context.params
  if (!ALLOWED_PROVIDER_CODES.has(providerCode)) {
    return Response.json({ error: "Unsupported provider code." }, { status: 400 })
  }

  let body: UpsertProviderKeyBody
  try {
    body = (await request.json()) as UpsertProviderKeyBody
  } catch {
    return Response.json({ error: "Invalid request body." }, { status: 400 })
  }

  const apiKey = body.apiKey?.trim() ?? ""
  const keyName = body.keyName?.trim() ?? "default"
  if (!apiKey) {
    return Response.json({ error: "API key is required." }, { status: 400 })
  }

  const payload = {
    api_key: apiKey,
    key_name: keyName,
    is_default: body.isDefault ?? true,
    is_active: body.isActive ?? true,
  }

  const backendEndpoint = new URL(
    `/users/dev/current/settings/api-keys/${providerCode}`,
    BACKEND_URL
  )

  try {
    const upstreamResponse = await fetch(backendEndpoint, {
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "PUT",
      body: JSON.stringify(payload),
      cache: "no-store",
    })

    if (!upstreamResponse.ok) {
      const errorText = await upstreamResponse.text()
      return Response.json(
        { error: errorText || "Backend provider key upsert failed." },
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
      { error: "Unable to reach provider keys backend." },
      { status: 502 }
    )
  }
}
