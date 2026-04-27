"use client";

import { createContext, useCallback, useContext, useState } from "react";
import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";

export type BackendProviderCode =
  | "openai"
  | "anthropic"
  | "gemini"
  | "groq"
  | "other";

export type CompareModelSelection = {
  modelId: string;
  providerCode: BackendProviderCode;
};

export type SubmitMessagePayload = {
  message: PromptInputMessage;
  modelId: string;
  providerCode: BackendProviderCode;
  compareWith?: CompareModelSelection[];
};

type ComposerHandlers = {
  onSubmitMessage: (payload: SubmitMessagePayload) => void | Promise<void>;
  onStop?: () => void;
  preferredModelId: string | null;
  forceModelId?: string | null;
};

type ChatComposerContextValue = {
  handlers: ComposerHandlers | null;
  register: (handlers: ComposerHandlers) => void;
  unregister: () => void;
};

const ChatComposerContext = createContext<ChatComposerContextValue | null>(null);

export function ChatComposerProvider({ children }: { children: React.ReactNode }) {
  const [handlers, setHandlers] = useState<ComposerHandlers | null>(null);

  const register = useCallback((h: ComposerHandlers) => {
    setHandlers(h);
  }, []);

  const unregister = useCallback(() => {
    setHandlers(null);
  }, []);

  return (
    <ChatComposerContext.Provider value={{ handlers, register, unregister }}>
      {children}
    </ChatComposerContext.Provider>
  );
}

export function useChatComposer() {
  const ctx = useContext(ChatComposerContext);
  if (!ctx) {
    throw new Error("useChatComposer must be used inside ChatComposerProvider");
  }
  return ctx;
}
