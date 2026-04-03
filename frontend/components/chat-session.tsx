"use client";

import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";
import type { FileUIPart } from "ai";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Attachment,
  AttachmentHoverCard,
  AttachmentHoverCardContent,
  AttachmentHoverCardTrigger,
  AttachmentPreview,
  Attachments,
} from "@/components/ai-elements/attachments";
import type {
  ConversationMessage,
  ConversationMessageCitation,
} from "@/components/ai-elements/conversation";
import { ModelSelectorLogo } from "@/components/ai-elements/model-selector";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { splitInlineCitations } from "@/lib/chat-citations";
import { toBackendChatAttachment } from "@/lib/chat-attachments";
import { FileTextIcon, ServerIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import type { BackendProviderCode } from "@/contexts/chat-composer-context";
import { useChatComposer } from "@/contexts/chat-composer-context";
import { useCallback, useEffect, useMemo, useState } from "react";

const normalizeMessageText = (message: PromptInputMessage) => message.text?.trim() ?? "";

const buildPendingReasoningLabel = (attachmentCount: number) =>
  attachmentCount > 0
    ? `Processing ${attachmentCount} file${attachmentCount === 1 ? "" : "s"}...`
    : "Thinking...";



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

interface MessageAttachmentsProps {
  attachments: FileUIPart[];
  align?: "start" | "end";
  messageIndex: number;
}

const MessageAttachments = ({
  attachments,
  align = "start",
  messageIndex,
}: MessageAttachmentsProps) => (
  <Attachments variant="grid" className={align === "start" ? "ml-0 mr-auto" : undefined}>
    {attachments.map((attachment, attachmentIndex) => {
      const attachmentId = `message-${messageIndex}-attachment-${attachmentIndex}`;
      const label = attachment.filename?.trim() || "Attachment";

      return (
        <AttachmentHoverCard key={attachmentId}>
          <AttachmentHoverCardTrigger>
            <Attachment
              data={{ ...attachment, id: attachmentId }}
              title={label}
            >
              <AttachmentPreview />
            </Attachment>
          </AttachmentHoverCardTrigger>
          <AttachmentHoverCardContent className="w-56">
            <div className="space-y-1">
              <div className="truncate font-medium text-sm">{label}</div>
              <div className="truncate text-muted-foreground text-xs">
                {attachment.mediaType}
              </div>
            </div>
          </AttachmentHoverCardContent>
        </AttachmentHoverCard>
      );
    })}
  </Attachments>
);

interface GroupedDocumentCitation {
  documentId: string;
  filename: string | null;
  chunks: ConversationMessageCitation[];
}

const groupCitationsByDocument = (
  citations: ConversationMessageCitation[]
): GroupedDocumentCitation[] => {
  const order: string[] = [];
  const byDocId = new Map<string, GroupedDocumentCitation>();
  for (const citation of citations) {
    if (!byDocId.has(citation.documentId)) {
      order.push(citation.documentId);
      byDocId.set(citation.documentId, {
        documentId: citation.documentId,
        filename: citation.filename ?? null,
        chunks: [],
      });
    }
    byDocId.get(citation.documentId)!.chunks.push(citation);
  }
  return order.map((id) => byDocId.get(id)!);
};

interface MessageCitationsProps {
  citations: ConversationMessageCitation[];
}

const MessageCitations = ({ citations }: MessageCitationsProps) => {
  const grouped = groupCitationsByDocument(citations);
  return (
    <TooltipProvider delay={300}>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {grouped.map((doc, docIndex) => {
          const label = doc.filename?.trim() || `Document ${docIndex + 1}`;
          return (
            <Tooltip key={doc.documentId}>
              <TooltipTrigger>
                <Badge
                  className="gap-1.5 rounded-full px-2 py-0.5 text-xs font-normal text-muted-foreground"
                  variant="outline"
                >
                  <FileTextIcon className="size-3 shrink-0" />
                  <span className="max-w-36 truncate">{label}</span>
                </Badge>
              </TooltipTrigger>
              <TooltipContent>{label}</TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </TooltipProvider>
  );
};



const ChatSession = ({ initialMessages, initialThreadId }: ChatSessionProps) => {
  const router = useRouter();
  const preferredModelId = useMemo(
    () => getLastUsedModelId(initialMessages),
    [initialMessages]
  );
  const [messages, setMessages] = useState<ConversationMessage[]>(() => initialMessages ?? []);
  const [threadId, setThreadId] = useState<string | null>(() => initialThreadId ?? null);
  const [hasThreadRoute, setHasThreadRoute] = useState<boolean>(() => Boolean(initialThreadId));

  useEffect(() => {
    if (!initialMessages) {
      return;
    }

    setMessages(initialMessages);
  }, [initialMessages]);

  useEffect(() => {
    setThreadId(initialThreadId ?? null);
    setHasThreadRoute(Boolean(initialThreadId));
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

  const updateAssistantMessageMeta = useCallback(
    (index: number, updates: Partial<ConversationMessage>) => {
      setMessages((prev) => {
        if (!prev[index]) {
          return prev;
        }

        const next = [...prev];
        next[index] = { ...next[index], ...updates };
        return next;
      });
    },
    []
  );

  const handleSubmitMessage = useCallback(async (payload: {
    message: PromptInputMessage;
    modelId: string;
    providerCode: BackendProviderCode;
  }) => {
    const userContent = normalizeMessageText(payload.message);
    const userAttachments = payload.message.files ?? [];
    const attachmentCount = userAttachments.length;
    const reasoningLabel = buildPendingReasoningLabel(attachmentCount);

    if (!userContent && userAttachments.length === 0) {
      return;
    }

    let assistantMessageIndex = -1;
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userContent, attachments: userAttachments },
      (() => {
        assistantMessageIndex = prev.length + 1;
        return {
          role: "assistant" as const,
          content: "",
          providerCode: payload.providerCode,
          modelName: payload.modelId,
          reasoningLabel,
          reasoningStreaming: true,
        };
      })(),
    ]);

    const shouldRedirectToThread = !hasThreadRoute;

    try {
      const response = await fetch("/api/chat", {
        body: JSON.stringify({
          attachments: userAttachments.map(toBackendChatAttachment),
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
        setHasThreadRoute(true);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";
      let hasReceivedFirstChunk = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        const raw = decoder.decode(value, { stream: true });
        const { text, citations } = splitInlineCitations(raw);
        assistantContent += text;

        if (citations.length > 0) {
          updateAssistantMessageMeta(assistantMessageIndex, { citations });
        }
        if (!hasReceivedFirstChunk) {
          hasReceivedFirstChunk = true;
          updateAssistantMessageMeta(assistantMessageIndex, {
            reasoningStreaming: false,
          });
        }
        updateAssistantMessage(assistantMessageIndex, assistantContent);
      }

      assistantContent += decoder.decode();
      updateAssistantMessageMeta(assistantMessageIndex, {
        reasoningStreaming: false,
      });
      updateAssistantMessage(
        assistantMessageIndex,
        assistantContent.trim() ? assistantContent : "(No response)"
      );

      // Sync the Next.js router with the URL set via replaceState during streaming.
      // Without this, Next.js still thinks the route is "/" and clicking "New Chat"
      // would be a same-route no-op that leaves the old messages visible.
      if (shouldRedirectToThread && resolvedThreadId) {
        router.replace(`/chats/${resolvedThreadId}`);
      }
      window.dispatchEvent(new Event("chat-threads-updated"));
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unexpected chat error.";

      updateAssistantMessageMeta(assistantMessageIndex, {
        reasoningStreaming: false,
      });
      updateAssistantMessage(assistantMessageIndex, `Error: ${errorMessage}`);
    }
  }, [hasThreadRoute, router, threadId, updateAssistantMessage, updateAssistantMessageMeta]);

  const { register, unregister } = useChatComposer();

  useEffect(() => {
    register({ onSubmitMessage: handleSubmitMessage, preferredModelId });
    return () => unregister();
  }, [register, unregister, handleSubmitMessage, preferredModelId]);

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden">
      <Conversation className="min-h-0 min-w-0 flex-1">
        <ConversationContent
          className={`mx-auto w-full max-w-3xl ${
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
              const metaText = message.modelName ? message.modelName : null;
              const trimmedContent = message.content.trim();
              const hasAttachments = Boolean(message.attachments?.length);
              const hasCitations = Boolean(message.citations?.length);

              return (
                <Message from={from} key={`${index}-${message.role}`}>
                  {hasAttachments ? (
                    <MessageAttachments
                      align={from === "assistant" ? "start" : "end"}
                      attachments={message.attachments ?? []}
                      messageIndex={index}
                    />
                  ) : null}
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
                        {message.reasoningStreaming ? (
                          <div className="text-muted-foreground mb-3 flex items-center gap-2 text-sm">
                            <span className="bg-current size-2 animate-pulse rounded-full" />
                            <Shimmer duration={1.2}>
                              {message.reasoningLabel ?? "Processing..."}
                            </Shimmer>
                          </div>
                        ) : null}
                        {metaText ? (
                          <div className="text-muted-foreground mb-1 text-xs">
                            {metaText}
                          </div>
                        ) : null}
                        {trimmedContent ? (
                          <MessageResponse>{message.content}</MessageResponse>
                        ) : null}
                        {hasCitations ? (
                          <MessageCitations citations={message.citations ?? []} />
                        ) : null}
                      </MessageContent>
                    </div>
                  ) : trimmedContent ? (
                    <MessageContent>
                      <MessageResponse>{message.content}</MessageResponse>
                    </MessageContent>
                  ) : null}
                </Message>
              );
            })
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
    </div>
  );
};

export default ChatSession;
