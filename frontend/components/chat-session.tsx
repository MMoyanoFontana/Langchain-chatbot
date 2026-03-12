"use client";

import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import type { ConversationMessage } from "@/components/ai-elements/conversation";
import { ModelSelectorLogo } from "@/components/ai-elements/model-selector";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import PromptComposer from "@/components/prompt-composer";
import { ServerIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

const buildMessageText = (message: PromptInputMessage) => {
  const text = message.text?.trim() ?? "";
  const attachmentNames =
    message.files
      ?.map((file) => file.filename)
      .filter((filename): filename is string => Boolean(filename)) ?? [];

  const attachmentLine =
    attachmentNames.length > 0
      ? `Attachments: ${attachmentNames.join(", ")}`
      : "";

  return [text, attachmentLine].filter(Boolean).join("\n");
};

interface ChatSessionProps {
  initialMessages?: ConversationMessage[];
  initialThreadId?: string;
}

const toProviderLogo = (providerCode: NonNullable<ConversationMessage["providerCode"]>) => {
  if (providerCode === "gemini") {
    return "google";
  }
  return providerCode;
};

const getLastUsedModelId = (messages?: ConversationMessage[]): string | null => {
  if (!messages?.length) {
    return null;
  }

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const modelName = messages[index].modelName?.trim();
    if (modelName) {
      return modelName;
    }
  }

  return null;
};

const ChatSession = ({ initialMessages, initialThreadId }: ChatSessionProps) => {
  const preferredModelId = useMemo(
    () => getLastUsedModelId(initialMessages),
    [initialMessages]
  );
  const [messages, setMessages] = useState<ConversationMessage[]>(() => initialMessages ?? []);
  const [threadId, setThreadId] = useState<string | null>(() => initialThreadId ?? null);

  useEffect(() => {
    if (!initialMessages) {
      return;
    }

    setMessages(initialMessages);
  }, [initialMessages]);

  useEffect(() => {
    setThreadId(initialThreadId ?? null);
  }, [initialThreadId]);

  const updateAssistantMessage = useCallback((index: number, content: string) => {
    setMessages((prev) => {
      if (!prev[index]) {
        return prev;
      }

      const next = [...prev];
      next[index] = { ...next[index], content };
      return next;
    });
  }, []);

  const handleSubmitMessage = useCallback(async (payload: {
    message: PromptInputMessage;
    modelId: string;
    providerCode: NonNullable<ConversationMessage["providerCode"]>;
  }) => {
    const userContent = buildMessageText(payload.message);

    if (!userContent) {
      return;
    }

    let assistantMessageIndex = -1;
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userContent },
      (() => {
        assistantMessageIndex = prev.length + 1;
        return {
          role: "assistant" as const,
          content: "",
          providerCode: payload.providerCode,
          modelName: payload.modelId,
        };
      })(),
    ]);

    const shouldRedirectToThread = !threadId;

    try {
      const response = await fetch("/api/chat", {
        body: JSON.stringify({
          prompt: userContent,
          threadId,
          modelId: payload.modelId,
          providerCode: payload.providerCode,
        }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Request failed with status ${response.status}`);
      }

      if (!response.body) {
        throw new Error("No response stream received.");
      }

      const resolvedThreadId = response.headers.get("x-thread-id");
      if (resolvedThreadId && resolvedThreadId !== threadId) {
        setThreadId(resolvedThreadId);
      }
      if (resolvedThreadId && shouldRedirectToThread) {
        // Keep component mounted so streamed assistant tokens remain visible
        // while still updating the URL to the thread route.
        window.history.replaceState(
          window.history.state,
          "",
          `/chats/${resolvedThreadId}`
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        assistantContent += decoder.decode(value, { stream: true });
        updateAssistantMessage(assistantMessageIndex, assistantContent);
      }

      assistantContent += decoder.decode();
      updateAssistantMessage(
        assistantMessageIndex,
        assistantContent.trim() ? assistantContent : "(No response)"
      );

      window.dispatchEvent(new Event("chat-threads-updated"));
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unexpected chat error.";

      updateAssistantMessage(assistantMessageIndex, `Error: ${errorMessage}`);
    }
  }, [threadId, updateAssistantMessage]);

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-xl bg-background">
      <Conversation className="min-h-0 min-w-0 flex-1">
        <ConversationContent
          className={`mx-auto w-full max-w-4xl ${
            messages.length === 0 ? "h-full justify-center" : "min-h-full"
          }`}
        >
          {messages.length === 0 ? (
            <ConversationEmptyState
              className="[&_h3]:text-2xl [&_h3]:font-semibold [&_p]:text-base"
              description="Ask me anything to get started. I am here to help."
              title="Ready when you are"
            />
          ) : (
            messages.map((message, index) => {
              const from =
                message.role === "user"
                  ? "user"
                  : message.role === "system"
                    ? "system"
                    : "assistant";
              const metaText = message.modelName
                  ? message.modelName
                  : null;

              return (
                <Message from={from} key={`${index}-${message.role}`}>
                  {from === "assistant" ? (
                    <div className="flex items-start gap-2.5">
                      <div className="bg-muted/40 border-border mt-0.5 flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full border">
                        {message.providerCode && message.providerCode !== "other" ? (
                          <ModelSelectorLogo
                            provider={toProviderLogo(message.providerCode)}
                            className="size-5"
                          />
                        ) : (
                          <ServerIcon className="text-muted-foreground size-5" />
                        )}
                      </div>
                      <MessageContent>
                        {metaText ? (
                          <div className="text-muted-foreground mb-1 text-xs">
                            {metaText}
                          </div>
                        ) : null}
                        <MessageResponse>{message.content}</MessageResponse>
                      </MessageContent>
                    </div>
                  ) : (
                    <MessageContent>
                      <MessageResponse>{message.content}</MessageResponse>
                    </MessageContent>
                  )}
                </Message>
              );
            })
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
      <div className="p-4">
        <PromptComposer
          className="mx-auto w-full max-w-4xl"
          preferredModelId={preferredModelId}
          onSubmitMessage={handleSubmitMessage}
        />
      </div>
    </div>
  );
};

export default ChatSession;
