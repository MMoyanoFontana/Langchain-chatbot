import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  parseUpstreamError,
  setSessionCookie,
} from "@/lib/backend-route";

const ALLOWED_AUTH_PROVIDERS = new Set(["google", "github", "microsoft"]);

const redirectToLogin = (request: NextRequest, error: string) => {
  const loginUrl = new URL("/login", request.nextUrl.origin);
  loginUrl.searchParams.set("error", error);
  return NextResponse.redirect(loginUrl);
};

const normalizeInternalRedirect = (value: string | null | undefined) => {
  const candidate = value?.trim() ?? "";
  if (!candidate.startsWith("/") || candidate.startsWith("//") || candidate.startsWith("/\\")) {
    return "/";
  }
  return candidate || "/";
};

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ provider: string }> }
) {
  const { provider } = await context.params;
  if (!ALLOWED_AUTH_PROVIDERS.has(provider)) {
    return redirectToLogin(request, "Unsupported sign-in provider.");
  }

  const code = request.nextUrl.searchParams.get("code")?.trim() ?? "";
  const state = request.nextUrl.searchParams.get("state")?.trim() ?? "";
  if (!code || !state) {
    return redirectToLogin(request, "Provider sign-in did not complete.");
  }

  const redirectUri = new URL(
    `/api/auth/oauth/${provider}/callback`,
    request.nextUrl.origin
  ).toString();

  try {
    const upstreamResponse = await backendFetchFromRoute(
      `/auth/oauth/${provider}/exchange`,
      {
        body: JSON.stringify({
          code,
          state,
          redirect_uri: redirectUri,
        }),
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        method: "POST",
      }
    );

    if (!upstreamResponse.ok) {
      return redirectToLogin(
        request,
        await parseUpstreamError(
          upstreamResponse,
          "Unable to complete provider sign-in."
        )
      );
    }

    const payload = (await upstreamResponse.json()) as {
      session_token: string;
      redirect_to?: string | null;
    };
    const destination = normalizeInternalRedirect(payload.redirect_to);
    const response = NextResponse.redirect(new URL(destination, request.nextUrl.origin));
    setSessionCookie(response, payload.session_token);
    return response;
  } catch {
    return redirectToLogin(request, "Unable to reach auth backend.");
  }
}
