import { NextResponse } from "next/server";

import {
  backendFetchFromRoute,
  clearSessionCookie,
  getRouteSessionToken,
} from "@/lib/backend-route";

export async function POST() {
  const sessionToken = await getRouteSessionToken();

  try {
    if (sessionToken) {
      await backendFetchFromRoute(
        "/auth/logout",
        {
          headers: { Accept: "application/json" },
          method: "POST",
        },
        sessionToken
      );
    }
  } catch {
    // Clear the local cookie even if the backend logout request fails.
  }

  const response = NextResponse.json({ ok: true });
  clearSessionCookie(response);
  return response;
}
