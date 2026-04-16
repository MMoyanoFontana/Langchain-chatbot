"use client";

import { useCallback, useMemo, useState } from "react";

import type { ThreadHistoryItem } from "@/hooks/use-thread-history";

export const MAX_THREAD_TITLE_LENGTH = 200;
const THREADS_UPDATED_EVENT = "chat-threads-updated";

const parseActionError = async (response: Response, fallback: string) => {
  try {
    const payload = (await response.json()) as {
      error?: string;
      detail?: string | { detail?: string };
    };

    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : typeof payload.detail?.detail === "string"
          ? payload.detail.detail
          : null;

    const candidate = payload.error ?? detail;
    return candidate?.trim() || fallback;
  } catch {
    try {
      const text = (await response.text()).trim();
      return text || fallback;
    } catch {
      return fallback;
    }
  }
};

type UseThreadActionsParams = {
  pathname: string;
  refreshHistory: () => Promise<void> | void;
  navigateToRoot: () => void;
};

export function useThreadActions({
  pathname,
  refreshHistory,
  navigateToRoot,
}: UseThreadActionsParams) {
  const [threadActionLoadingId, setThreadActionLoadingId] = useState<string | null>(null);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [systemPromptDialogOpen, setSystemPromptDialogOpen] = useState(false);
  const [activeThread, setActiveThread] = useState<ThreadHistoryItem | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [systemPromptValue, setSystemPromptValue] = useState("");
  const [systemPromptError, setSystemPromptError] = useState<string | null>(null);

  const normalizedRenameValue = renameValue.trim();
  const renameUnchanged = activeThread
    ? normalizedRenameValue === activeThread.title
    : false;
  const renameValidationError = !renameDialogOpen || !activeThread
    ? null
    : normalizedRenameValue.length === 0
      ? "Name cannot be blank."
      : normalizedRenameValue.length > MAX_THREAD_TITLE_LENGTH
        ? `Name must be ${MAX_THREAD_TITLE_LENGTH} characters or fewer.`
        : null;

  const canSubmitRename = useMemo(
    () =>
      Boolean(activeThread) &&
      !renameValidationError &&
      !renameUnchanged &&
      !threadActionLoadingId,
    [activeThread, renameValidationError, renameUnchanged, threadActionLoadingId]
  );

  const canSubmitDelete = useMemo(
    () => Boolean(activeThread) && !threadActionLoadingId,
    [activeThread, threadActionLoadingId]
  );

  const dispatchThreadUpdate = useCallback(() => {
    window.dispatchEvent(new Event(THREADS_UPDATED_EVENT));
  }, []);

  const openRenameDialog = useCallback((thread: ThreadHistoryItem) => {
    setActiveThread(thread);
    setRenameValue(thread.title);
    setRenameError(null);
    setRenameDialogOpen(true);
  }, []);

  const openDeleteDialog = useCallback((thread: ThreadHistoryItem) => {
    setActiveThread(thread);
    setDeleteError(null);
    setDeleteDialogOpen(true);
  }, []);

  const closeRenameDialog = useCallback(() => {
    if (threadActionLoadingId) {
      return;
    }

    setRenameDialogOpen(false);
    setRenameError(null);
    setActiveThread(null);
    setRenameValue("");
  }, [threadActionLoadingId]);

  const closeDeleteDialog = useCallback(() => {
    if (threadActionLoadingId) {
      return;
    }

    setDeleteDialogOpen(false);
    setDeleteError(null);
    setActiveThread(null);
  }, [threadActionLoadingId]);

  const openSystemPromptDialog = useCallback((thread: ThreadHistoryItem) => {
    setActiveThread(thread);
    setSystemPromptValue(thread.systemPrompt ?? "");
    setSystemPromptError(null);
    setSystemPromptDialogOpen(true);
  }, []);

  const closeSystemPromptDialog = useCallback(() => {
    if (threadActionLoadingId) return;
    setSystemPromptDialogOpen(false);
    setSystemPromptError(null);
    setActiveThread(null);
    setSystemPromptValue("");
  }, [threadActionLoadingId]);

  const submitSystemPrompt = useCallback(async () => {
    if (!activeThread) return;

    setThreadActionLoadingId(activeThread.id);
    setSystemPromptError(null);

    try {
      const response = await fetch(`/api/threads/${activeThread.id}/system-prompt`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ system_prompt: systemPromptValue.trim() || null }),
      });

      if (!response.ok) {
        const message = await parseActionError(response, "Failed to update system prompt.");
        throw new Error(message);
      }

      closeSystemPromptDialog();
      await refreshHistory();
      dispatchThreadUpdate();
    } catch (error) {
      setSystemPromptError(
        error instanceof Error ? error.message : "Could not update system prompt."
      );
    } finally {
      setThreadActionLoadingId(null);
    }
  }, [activeThread, closeSystemPromptDialog, dispatchThreadUpdate, refreshHistory, systemPromptValue]);

  const submitRename = useCallback(async () => {
    if (!activeThread) {
      return;
    }

    if (renameValidationError) {
      setRenameError(renameValidationError);
      return;
    }

    if (renameUnchanged) {
      return;
    }

    const title = normalizedRenameValue;
    setThreadActionLoadingId(activeThread.id);
    setRenameError(null);

    try {
      const response = await fetch(`/api/threads/${activeThread.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });

      if (!response.ok) {
        const message = await parseActionError(response, "Failed to rename chat.");
        throw new Error(message);
      }

      closeRenameDialog();
      await refreshHistory();
      dispatchThreadUpdate();
    } catch (error) {
      setRenameError(
        error instanceof Error ? error.message : "Could not rename this chat."
      );
    } finally {
      setThreadActionLoadingId(null);
    }
  }, [
    activeThread,
    closeRenameDialog,
    dispatchThreadUpdate,
    normalizedRenameValue,
    refreshHistory,
    renameUnchanged,
    renameValidationError,
  ]);

  const submitDelete = useCallback(async () => {
    if (!activeThread) {
      return;
    }

    setThreadActionLoadingId(activeThread.id);
    setDeleteError(null);

    try {
      const response = await fetch(`/api/threads/${activeThread.id}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const message = await parseActionError(response, "Failed to delete chat.");
        throw new Error(message);
      }

      if (pathname === `/chats/${activeThread.id}`) {
        navigateToRoot();
      }

      closeDeleteDialog();
      await refreshHistory();
      dispatchThreadUpdate();
    } catch (error) {
      setDeleteError(
        error instanceof Error ? error.message : "Could not delete this chat."
      );
    } finally {
      setThreadActionLoadingId(null);
    }
  }, [
    activeThread,
    closeDeleteDialog,
    dispatchThreadUpdate,
    navigateToRoot,
    pathname,
    refreshHistory,
  ]);

  return {
    threadActionLoadingId,
    renameDialogOpen,
    deleteDialogOpen,
    systemPromptDialogOpen,
    activeThread,
    renameValue,
    renameError,
    deleteError,
    systemPromptValue,
    systemPromptError,
    renameValidationError,
    canSubmitRename,
    canSubmitDelete,
    setRenameValue,
    setSystemPromptValue,
    openRenameDialog,
    openDeleteDialog,
    openSystemPromptDialog,
    closeRenameDialog,
    closeDeleteDialog,
    closeSystemPromptDialog,
    submitRename,
    submitDelete,
    submitSystemPrompt,
  };
}
