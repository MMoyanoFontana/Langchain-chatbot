"use client";

import { useCallback, useEffect, useState } from "react";

export type SidebarUser = {
  name: string;
  email: string;
  avatar: string | null;
};

type CurrentUserResponse = {
  email?: string | null;
  full_name: string | null;
  avatar_url?: string | null;
};

const toSidebarUser = (payload: CurrentUserResponse): SidebarUser | null => {
  const email = payload.email?.trim() ?? "";
  if (!email) {
    return null;
  }

  return {
    name: payload.full_name?.trim() || email,
    email,
    avatar: payload.avatar_url?.trim() || null,
  };
};

export function useSidebarUser(initialUser?: SidebarUser | null) {
  const [user, setUser] = useState<SidebarUser | null>(initialUser ?? null);
  const [isLoading, setIsLoading] = useState(initialUser === undefined);

  const refresh = useCallback(async (showLoader = true) => {
    if (showLoader) {
      setIsLoading(true);
    }

    try {
      const response = await fetch("/api/user", { cache: "no-store" });
      if (response.status === 401 || response.status === 404) {
        setUser(null);
        return;
      }

      if (!response.ok) {
        return;
      }

      const payload = (await response.json()) as CurrentUserResponse;
      setUser(toSidebarUser(payload));
    } catch {
      // Keep current state if network is unavailable.
    } finally {
      if (showLoader) {
        setIsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (initialUser !== undefined) {
      setUser(initialUser);
      setIsLoading(false);
    }
  }, [initialUser]);

  useEffect(() => {
    void refresh(initialUser === undefined);
  }, [initialUser, refresh]);

  return {
    user,
    isLoading,
    refresh,
  };
}
