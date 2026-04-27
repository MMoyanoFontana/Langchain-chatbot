import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  parseUpstreamError,
  setSessionCookie,
} from "@/lib/backend-route";

type LoginBody = {
  username?: string;
  password?: string;
};

export async function POST(request: NextRequest) {
  let body: LoginBody;
  try {
    body = (await request.json()) as LoginBody;
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  const username = body.username?.trim() ?? "";
  const password = body.password ?? "";
  if (!username || !password) {
    return NextResponse.json({ error: "Username and password are required." }, { status: 400 });
  }

  try {
    const upstream = await backendFetchFromRoute("/auth/login", {
      body: JSON.stringify({ username, password }),
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      method: "POST",
    });

    if (!upstream.ok) {
      return NextResponse.json(
        { error: await parseUpstreamError(upstream, "Unable to sign in.") },
        { status: upstream.status || 502 }
      );
    }

    const payload = (await upstream.json()) as { session_token: string; user: unknown };
    const response = NextResponse.json({ user: payload.user });
    setSessionCookie(response, payload.session_token);
    return response;
  } catch {
    return NextResponse.json({ error: "Unable to reach auth backend." }, { status: 502 });
  }
}
