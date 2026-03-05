const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"
const ALLOWED_PROVIDER_CODES = new Set(["openai", "anthropic", "gemini", "groq", "xai", "openrouter", "other"])

type ProviderApiKeyRead = {
  id: string
  key_name: string
  provider: {
    code: string
  }
}

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

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ providerCode: string; keyName: string }> }
) {
  const { providerCode, keyName } = await context.params
  if (!ALLOWED_PROVIDER_CODES.has(providerCode)) {
    return Response.json({ error: "Unsupported provider code." }, { status: 400 })
  }

  const listEndpoint = new URL("/users/dev/current/settings/api-keys", BACKEND_URL)

  try {
    const listResponse = await fetch(listEndpoint, {
      headers: { Accept: "application/json" },
      method: "GET",
      cache: "no-store",
    })

    if (!listResponse.ok) {
      return Response.json(
        { error: await parseUpstreamError(listResponse, "Backend provider keys request failed.") },
        { status: listResponse.status || 502 }
      )
    }

    const keys = (await listResponse.json()) as ProviderApiKeyRead[]
    const keyToDelete = keys.find(
      (entry) =>
        entry.provider?.code === providerCode && entry.key_name === keyName
    )

    if (!keyToDelete) {
      return new Response(null, { status: 204 })
    }

    const deleteEndpoint = new URL(
      `/users/dev/current/settings/api-keys/${keyToDelete.id}`,
      BACKEND_URL
    )

    const deleteResponse = await fetch(deleteEndpoint, {
      headers: { Accept: "application/json" },
      method: "DELETE",
      cache: "no-store",
    })

    if (!deleteResponse.ok) {
      return Response.json(
        { error: await parseUpstreamError(deleteResponse, "Backend provider key delete failed.") },
        { status: deleteResponse.status || 502 }
      )
    }

    return new Response(null, { status: 204 })
  } catch {
    return Response.json(
      { error: "Unable to reach provider keys backend." },
      { status: 502 }
    )
  }
}
