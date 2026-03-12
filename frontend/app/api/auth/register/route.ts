import { NextRequest, NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  parseUpstreamError,
  setSessionCookie,
} from "@/lib/backend-route";

type RegisterBody = {
  email?: string;
  password?: string;
  fullName?: string | null;
};

export async function POST(request: NextRequest) {
  let body: RegisterBody;

  try {
    body = (await request.json()) as RegisterBody;
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  const email = body.email?.trim() ?? "";
  const password = body.password ?? "";
  const fullName = body.fullName?.trim() ?? "";
  if (!email || !password) {
    return NextResponse.json(
      { error: "Email and password are required." },
      { status: 400 }
    );
  }

  try {
    const upstreamResponse = await backendFetchFromRoute("/auth/email/register", {
      body: JSON.stringify({
        email,
        password,
        full_name: fullName || null,
      }),
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      method: "POST",
    });

    if (!upstreamResponse.ok) {
      return NextResponse.json(
        {
          error: await parseUpstreamError(
            upstreamResponse,
            "Unable to create account."
          ),
        },
        { status: upstreamResponse.status || 502 }
      );
    }

    const payload = (await upstreamResponse.json()) as {
      session_token: string;
      user: unknown;
    };
    const response = NextResponse.json({ user: payload.user });
    setSessionCookie(response, payload.session_token);
    return response;
  } catch {
    return NextResponse.json(
      { error: "Unable to reach auth backend." },
      { status: 502 }
    );
  }
}
