import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  normalizeInternalRedirect,
  parseUpstreamError,
} from "@/lib/backend-route";

const ALLOWED_AUTH_PROVIDERS = new Set(["google", "github", "microsoft"]);
const OAUTH_NONCE_COOKIE = "oauth_nonce";
const OAUTH_NONCE_MAX_AGE = 600;

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
        await parseUpstreamError(upstreamResponse, "Unable to start provider sign-in.")
      );
    }

    const payload = (await upstreamResponse.json()) as { authorize_url: string };

    // Extract nonce from backend Set-Cookie and forward it to the browser
    const nonce = extractNonceFromSetCookie(upstreamResponse.headers.get("set-cookie"));

    const response = NextResponse.redirect(new URL(payload.authorize_url));
    if (nonce) {
      response.cookies.set(OAUTH_NONCE_COOKIE, nonce, {
        httpOnly: true,
        maxAge: OAUTH_NONCE_MAX_AGE,
        path: "/",
        sameSite: "lax",
        secure: process.env.NODE_ENV === "production",
      });
    }
    return response;
  } catch {
    return redirectToAuthPage(request, "Unable to reach auth backend.");
  }
}

function extractNonceFromSetCookie(header: string | null): string | null {
  if (!header) return null;
  for (const part of header.split(";")) {
    const trimmed = part.trim();
    if (trimmed.startsWith(`${OAUTH_NONCE_COOKIE}=`)) {
      return trimmed.slice(OAUTH_NONCE_COOKIE.length + 1);
    }
  }
  return null;
}
