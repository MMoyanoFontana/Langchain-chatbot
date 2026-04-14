"use client";

import { ChatComposerProvider, useChatComposer } from "@/contexts/chat-composer-context";
import PromptComposer from "@/components/prompt-composer";

function ComposerSlot() {
  const { handlers } = useChatComposer();
  return (
    <div className="p-4">
      <PromptComposer
        className="mx-auto w-full max-w-3xl"
        preferredModelId={handlers?.preferredModelId ?? null}
        forceModelId={handlers?.forceModelId ?? null}
        onSubmitMessage={handlers?.onSubmitMessage}
      />
    </div>
  );
}

export function ChatShell({ children }: { children: React.ReactNode }) {
  return (
    <ChatComposerProvider>
      <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-xl bg-background">
        <div className="min-h-0 min-w-0 flex-1 overflow-hidden">
          {children}
        </div>
        <ComposerSlot />
      </div>
    </ChatComposerProvider>
  );
}
