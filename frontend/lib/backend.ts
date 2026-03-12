import { cookies } from "next/headers";

export const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";
export const AUTH_SESSION_COOKIE_NAME = "langchain_chatbot_session";
export const AUTH_SESSION_MAX_AGE = 60 * 60 * 24 * 30;

export type BackendCurrentUser = {
  id: string;
  email: string;
  full_name: string | null;
  avatar_url?: string | null;
};

export const buildBackendUrl = (path: string) => new URL(path, BACKEND_URL);

export const buildBackendHeaders = (
  sessionToken: string | null,
  headers?: HeadersInit
) => {
  const nextHeaders = new Headers(headers);
  if (sessionToken) {
    nextHeaders.set("Authorization", `Bearer ${sessionToken}`);
  }
  return nextHeaders;
};

export const fetchBackend = async (
  path: string,
  init: RequestInit = {},
  sessionToken?: string | null
) =>
  fetch(buildBackendUrl(path), {
    ...init,
    cache: init.cache ?? "no-store",
    headers: buildBackendHeaders(sessionToken ?? null, init.headers),
  });

export const getServerSessionToken = async () =>
  (await cookies()).get(AUTH_SESSION_COOKIE_NAME)?.value?.trim() ?? null;

export const getServerCurrentUser = async (): Promise<BackendCurrentUser | null> => {
  const sessionToken = await getServerSessionToken();
  if (!sessionToken) {
    return null;
  }

  try {
    const response = await fetchBackend(
      "/auth/session",
      {
        headers: { Accept: "application/json" },
      },
      sessionToken
    );

    if (!response.ok) {
      if (process.env.NODE_ENV !== "production") {
        console.warn("[auth] /auth/session returned", response.status);
      }
      return null;
    }

    return (await response.json()) as BackendCurrentUser;
  } catch (error) {
    if (process.env.NODE_ENV !== "production") {
      console.warn("[auth] /auth/session request failed", error);
    }
    return null;
  }
};
