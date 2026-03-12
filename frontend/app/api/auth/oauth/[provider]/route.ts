import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  normalizeInternalRedirect,
  parseUpstreamError,
} from "@/lib/backend-route";

const ALLOWED_AUTH_PROVIDERS = new Set(["google", "github", "microsoft"]);

const redirectToAuthPage = (request: NextRequest, error: string) => {
  const loginUrl = new URL("/login", request.nextUrl.origin);
  loginUrl.searchParams.set("error", error);
  return NextResponse.redirect(loginUrl);
};

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ provider: string }> }
) {
  const { provider } = await context.params;
  if (!ALLOWED_AUTH_PROVIDERS.has(provider)) {
    return redirectToAuthPage(request, "Unsupported sign-in provider.");
  }

  const redirectUri = new URL(
    `/api/auth/oauth/${provider}/callback`,
    request.nextUrl.origin
  ).toString();
  const returnTo = normalizeInternalRedirect(request.nextUrl.searchParams.get("returnTo"));

  try {
    const upstreamResponse = await backendFetchFromRoute(
      `/auth/oauth/${provider}/authorize?${new URLSearchParams({
        redirect_uri: redirectUri,
        return_to: returnTo,
      }).toString()}`,
      {
        headers: { Accept: "application/json" },
        method: "GET",
      }
    );

    if (!upstreamResponse.ok) {
      return redirectToAuthPage(
        request,
        await parseUpstreamError(
          upstreamResponse,
          "Unable to start provider sign-in."
        )
      );
    }

    const payload = (await upstreamResponse.json()) as {
      authorize_url: string;
    };
    return NextResponse.redirect(new URL(payload.authorize_url));
  } catch {
    return redirectToAuthPage(request, "Unable to reach auth backend.");
  }
}
