import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import {
  AUTH_SESSION_COOKIE_NAME,
  AUTH_SESSION_MAX_AGE,
  buildBackendHeaders,
  buildBackendUrl,
} from "@/lib/backend";

export const getRouteSessionToken = async () =>
  (await cookies()).get(AUTH_SESSION_COOKIE_NAME)?.value?.trim() ?? null;

export const backendFetchFromRoute = async (
  path: string,
  init: RequestInit = {},
  sessionToken?: string | null
) =>
  fetch(buildBackendUrl(path), {
    ...init,
    cache: init.cache ?? "no-store",
    headers: buildBackendHeaders(sessionToken ?? null, init.headers),
  });

export const parseUpstreamError = async (response: Response, fallback: string) => {
  try {
    const payload = (await response.json()) as {
      detail?: string | { detail?: string };
      error?: string;
    };

    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : typeof payload.detail?.detail === "string"
          ? payload.detail.detail
          : null;

    return (payload.error ?? detail ?? fallback).trim();
  } catch {
    try {
      const errorText = (await response.text()).trim();
      return errorText || fallback;
    } catch {
      return fallback;
    }
  }
};

export const setSessionCookie = (response: NextResponse, sessionToken: string) => {
  response.cookies.set(AUTH_SESSION_COOKIE_NAME, sessionToken, {
    httpOnly: true,
    maxAge: AUTH_SESSION_MAX_AGE,
    path: "/",
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
  });
};

export const clearSessionCookie = (response: NextResponse) => {
  response.cookies.set(AUTH_SESSION_COOKIE_NAME, "", {
    expires: new Date(0),
    httpOnly: true,
    path: "/",
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
  });
};
