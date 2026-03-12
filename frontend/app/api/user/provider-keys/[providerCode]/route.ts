import { NextRequest, NextResponse } from "next/server";

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

type UpsertProviderKeyBody = {
  apiKey?: string;
  keyName?: string;
  isDefault?: boolean;
  isActive?: boolean;
};

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ providerCode: string }> }
) {
  const sessionToken = await getRouteSessionToken();
  if (!sessionToken) {
    return NextResponse.json({ error: "Authentication is required." }, { status: 401 });
  }

  const { providerCode } = await context.params;
  if (!ALLOWED_PROVIDER_CODES.has(providerCode)) {
    return NextResponse.json({ error: "Unsupported provider code." }, { status: 400 });
  }

  let body: UpsertProviderKeyBody;
  try {
    body = (await request.json()) as UpsertProviderKeyBody;
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  const apiKey = body.apiKey?.trim() ?? "";
  const keyName = body.keyName?.trim() ?? "default";
  if (!apiKey) {
    return NextResponse.json({ error: "API key is required." }, { status: 400 });
  }

  try {
    const upstreamResponse = await backendFetchFromRoute(
      `/users/me/settings/api-keys/${providerCode}`,
      {
        body: JSON.stringify({
          api_key: apiKey,
          key_name: keyName,
          is_default: body.isDefault ?? true,
          is_active: body.isActive ?? true,
        }),
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        method: "PUT",
      },
      sessionToken
    );

    if (!upstreamResponse.ok) {
      return NextResponse.json(
        {
          error: await parseUpstreamError(
            upstreamResponse,
            "Backend provider key upsert failed."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = await upstreamResponse.json();
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
    });
  } catch {
    return NextResponse.json(
      { error: "Unable to reach provider keys backend." },
      { status: 502 }
    );
  }
}
