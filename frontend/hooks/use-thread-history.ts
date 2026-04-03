"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type ThreadSummaryResponse = {
  id: string;
  title: string | null;
};

export type ThreadHistoryItem = {
  id: string;
  title: string;
};

const THREADS_UPDATED_EVENT = "chat-threads-updated";

export function useThreadHistory(refreshKey?: string) {
  const [threads, setThreads] = useState<ThreadHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const hasLoaded = useRef(false);

  const refresh = useCallback(async () => {
    setError(null);
    // Only show the loading spinner before the first successful fetch.
    // Subsequent refreshes update silently so the list doesn't flicker.
    if (!hasLoaded.current) {
      setIsLoading(true);
    }

    try {
      const response = await fetch("/api/threads", { cache: "no-store" });

      if (response.status === 401 || response.status === 404) {
        setThreads([]);
        return;
      }

      if (!response.ok) {
        throw new Error("Failed to load chats.");
      }

      const payload = (await response.json()) as ThreadSummaryResponse[];
      hasLoaded.current = true;
      setThreads(
        payload.map((thread) => ({
          id: thread.id,
          title: thread.title?.trim() || "Untitled chat",
        }))
      );
    } catch {
      setThreads([]);
      setError("Could not load chats.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh, refreshKey]);

  useEffect(() => {
    const handleThreadUpdate = () => {
      void refresh();
    };

    window.addEventListener(THREADS_UPDATED_EVENT, handleThreadUpdate);
    return () => {
      window.removeEventListener(THREADS_UPDATED_EVENT, handleThreadUpdate);
    };
  }, [refresh]);

  return {
    threads,
    isLoading,
    error,
    refresh,
  };
}
