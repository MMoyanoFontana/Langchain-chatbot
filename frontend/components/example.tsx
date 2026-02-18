"use client";

import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import type { ConversationMessage } from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import PromptComposer from "@/components/prompt-composer";
import { MessageSquareIcon } from "lucide-react";
import { useCallback, useState } from "react";

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

const ConversationDemo = () => {
  const [messages, setMessages] = useState<ConversationMessage[]>([]);

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

  const handleSubmitMessage = useCallback(async (message: PromptInputMessage) => {
    const userContent = buildMessageText(message);

    if (!userContent) {
      return;
    }

    let assistantMessageIndex = -1;
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userContent },
      (() => {
        assistantMessageIndex = prev.length + 1;
        return { role: "assistant" as const, content: "" };
      })(),
    ]);

    try {
      const response = await fetch("/api/chat", {
        body: JSON.stringify({ prompt: userContent }),
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
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unexpected chat error.";

      updateAssistantMessage(assistantMessageIndex, `Error: ${errorMessage}`);
    }
  }, [updateAssistantMessage]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl bg-background">
      <Conversation className="min-h-0 flex-1">
        <ConversationContent className="min-h-full">
          {messages.length === 0 ? (
            <ConversationEmptyState
              description="Send a prompt below to start the chat."
              icon={<MessageSquareIcon className="size-5" />}
              title="No messages yet"
            />
          ) : (
            messages.map((message, index) => {
              const from =
                message.role === "user"
                  ? "user"
                  : message.role === "system"
                    ? "system"
                    : "assistant";

              return (
                <Message from={from} key={`${index}-${message.role}`}>
                  <MessageContent>{message.content}</MessageContent>
                </Message>
              );
            })
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
      <div className="p-4">
        <PromptComposer onSubmitMessage={handleSubmitMessage} />
      </div>
    </div>
  );
};

export default ConversationDemo;
