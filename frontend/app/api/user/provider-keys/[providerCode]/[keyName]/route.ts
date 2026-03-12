import { NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  getRouteSessionToken,
  parseUpstreamError,
} from "@/lib/backend-route";

const ALLOWED_PROVIDER_CODES = new Set([
  "openai",
  "anthropic",
  "gemini",
  "groq",
  "xai",
  "openrouter",
  "other",
]);

type ProviderApiKeyRead = {
  id: string;
  key_name: string;
  provider: {
    code: string;
  };
};

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ providerCode: string; keyName: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { providerCode, keyName } = await context.params;
  if (!ALLOWED_PROVIDER_CODES.has(providerCode)) {
    return NextResponse.json({ error: "Unsupported provider code." }, { status: 400 });
  }

  try {
    const listResponse = await backendFetchFromRoute(
      "/users/me/settings/api-keys",
      {
        headers: { Accept: "application/json" },
        method: "GET",
      },
      sessionToken
    );

    if (!listResponse.ok) {
      return NextResponse.json(
        {
          error: await parseUpstreamError(
            listResponse,
            "Backend provider keys request failed."
          ),
        },
        { status: listResponse.status || 502 }
      );
    }

    const keys = (await listResponse.json()) as ProviderApiKeyRead[];
    const keyToDelete = keys.find(
      (entry) =>
        entry.provider?.code === providerCode && entry.key_name === keyName
    );

    if (!keyToDelete) {
      return new Response(null, { status: 204 });
    }

    const deleteResponse = await backendFetchFromRoute(
      `/users/me/settings/api-keys/${keyToDelete.id}`,
      {
        headers: { Accept: "application/json" },
        method: "DELETE",
      },
      sessionToken
    );

    if (!deleteResponse.ok) {
      return NextResponse.json(
        {
          error: await parseUpstreamError(
            deleteResponse,
            "Backend provider key delete failed."
          ),
        },
        { status: deleteResponse.status || 502 }
      );
    }

    return new Response(null, { status: 204 });
  } catch {
    return NextResponse.json(
      { error: "Unable to reach provider keys backend." },
      { status: 502 }
    );
  }
}
