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
  ConversationMessageMetrics,
} from "@/components/ai-elements/conversation";
import { ModelSelectorLogo } from "@/components/ai-elements/model-selector";
import {
  Message,
  MessageAction,
  MessageActions,
  MessageBranch,
  MessageBranchContent,
  MessageBranchNext,
  MessageBranchPage,
  MessageBranchPrevious,
  MessageBranchSelector,
  MessageContent,
  MessageResponse,
  MessageToolbar,
} from "@/components/ai-elements/message";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { SystemPromptDialog } from "@/components/system-prompt-dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { splitInlineCitations } from "@/lib/chat-citations";
import { toBackendChatAttachment } from "@/lib/chat-attachments";
import { BotIcon, CheckCircle2Icon, CheckIcon, CopyIcon, ExternalLinkIcon, FileTextIcon, Maximize2Icon, RefreshCwIcon, ServerIcon, Settings2Icon } from "lucide-react";
import { useRouter } from "next/navigation";
import type {
  BackendProviderCode,
  CompareModelSelection,
} from "@/contexts/chat-composer-context";
import { useChatComposer } from "@/contexts/chat-composer-context";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

const normalizeMessageText = (message: PromptInputMessage) => message.text?.trim() ?? "";

const buildPendingReasoningLabel = (attachmentCount: number) =>
  attachmentCount > 0
    ? `Processing ${attachmentCount} file${attachmentCount === 1 ? "" : "s"}...`
    : "Thinking...";



interface ChatSessionProps {
  initialMessages?: ConversationMessage[];
  initialThreadId?: string;
  initialSystemPrompt?: string | null;
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

const MessageAttachments = memo(({
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
));

MessageAttachments.displayName = "MessageAttachments";

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

const MessageCitations = memo(({ citations }: MessageCitationsProps) => {
  const grouped = useMemo(() => groupCitationsByDocument(citations), [citations]);
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
});

MessageCitations.displayName = "MessageCitations";

const formatLatency = (ms: number | null | undefined): string | null => {
  if (typeof ms !== "number" || !Number.isFinite(ms) || ms < 0) return null;
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
};

const formatTokens = (n: number | null | undefined): string | null => {
  if (typeof n !== "number" || !Number.isFinite(n) || n < 0) return null;
  return n.toLocaleString();
};

const computeTokensPerSecond = (
  metrics: ConversationMessageMetrics | null | undefined
): string | null => {
  if (!metrics) return null;
  const completion = metrics.completionTokens;
  const latency = metrics.latencyMs;
  const ttft = metrics.timeToFirstTokenMs ?? 0;
  if (
    typeof completion !== "number" ||
    typeof latency !== "number" ||
    completion <= 0 ||
    latency <= 0
  ) {
    return null;
  }
  const generationMs = Math.max(latency - ttft, 1);
  const tps = (completion / generationMs) * 1000;
  if (!Number.isFinite(tps) || tps <= 0) return null;
  return `${tps.toFixed(tps < 10 ? 1 : 0)} t/s`;
};

interface MessageMetricsProps {
  metrics: ConversationMessageMetrics | null | undefined;
}

const MessageMetrics = memo(({ metrics }: MessageMetricsProps) => {
  if (!metrics) return null;
  const items: { label: string; value: string }[] = [];
  const latency = formatLatency(metrics.latencyMs);
  if (latency) items.push({ label: "latency", value: latency });
  const ttft = formatLatency(metrics.timeToFirstTokenMs);
  if (ttft) items.push({ label: "ttft", value: ttft });
  const promptTokens = formatTokens(metrics.promptTokens);
  const completionTokens = formatTokens(metrics.completionTokens);
  if (promptTokens || completionTokens) {
    items.push({
      label: "tokens",
      value: `${promptTokens ?? "?"} in / ${completionTokens ?? "?"} out`,
    });
  }
  const tps = computeTokensPerSecond(metrics);
  if (tps) items.push({ label: "speed", value: tps });

  if (items.length === 0) return null;

  return (
    <div className="text-muted-foreground mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs">
      {items.map((item) => (
        <span key={item.label}>
          <span className="opacity-70">{item.label}:</span>{" "}
          <span className="font-medium">{item.value}</span>
        </span>
      ))}
    </div>
  );
});

MessageMetrics.displayName = "MessageMetrics";

interface ExpandMessageButtonProps {
  message: ConversationMessage;
}

const ExpandMessageButton = ({ message }: ExpandMessageButtonProps) => {
  const [open, setOpen] = useState(false);
  const title = message.modelName?.trim() || "Message";

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <MessageAction
        tooltip="Expand"
        onClick={() => setOpen(true)}
      >
        <Maximize2Icon className="size-3.5" />
      </MessageAction>
      <DialogContent
        className="flex max-h-[85vh] w-full max-w-3xl flex-col gap-0 p-0 sm:max-w-3xl"
      >
        <DialogHeader className="shrink-0 border-b px-5 py-3.5">
          <DialogTitle className="flex items-center gap-2 text-sm font-medium">
            {message.providerCode && message.providerCode !== "other" ? (
              <ModelSelectorLogo
                provider={toProviderLogo(message.providerCode)}
                className="size-4"
              />
            ) : (
              <ServerIcon className="text-muted-foreground size-4" />
            )}
            {title}
          </DialogTitle>
        </DialogHeader>
        <ScrollArea className="min-h-0 flex-1 overflow-auto">
          <div className="w-full px-5 py-4">
            <MessageResponse className="text-sm [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_table]:max-w-full [&_table]:overflow-x-auto">{message.content}</MessageResponse>
            {message.citations?.length ? (
              <MessageCitations citations={message.citations} />
            ) : null}
            {message.metrics ? (
              <MessageMetrics metrics={message.metrics} />
            ) : null}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};

interface CopyButtonProps {
  text: string;
}

const CopyButton = ({ text }: CopyButtonProps) => {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleCopy = useCallback(() => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => setCopied(false), 2000);
    });
  }, [text]);

  return (
    <MessageAction
      tooltip={copied ? "Copied" : "Copy"}
      onClick={handleCopy}
    >
      {copied ? <CheckIcon className="size-3.5" /> : <CopyIcon className="size-3.5" />}
    </MessageAction>
  );
};


type BranchEntry = {
  message: ConversationMessage;
  flatIndex: number;
};

type DisplayEntry =
  | { type: "single"; message: ConversationMessage; flatIndex: number }
  | { type: "branches"; branches: BranchEntry[]; parentId: string };

/**
 * Build a tree-walk display list from the flat message array.
 *
 * `branchSelections` maps a branch-point parentId to the selected branch index.
 * The algorithm:
 *  1. Infer parent links for legacy messages that lack parentMessageId.
 *  2. Build lookup maps: children-by-parent, message-by-id.
 *  3. Find the root (first message with no parent) and walk the selected path,
 *     grouping assistant siblings at each branch point.
 */
const buildTreeDisplay = (
  messages: ConversationMessage[],
  branchSelections: Map<string, number>,
): DisplayEntry[] => {
  if (messages.length === 0) return [];

  // Assign stable local IDs and infer parent chains for messages without them.
  const withIds = messages.map((msg, i) => ({
    ...msg,
    _stableId: msg.id ?? `local-${i}`,
    _flatIndex: i,
  }));

  // Infer parent links for legacy messages: walk linearly, linking each message
  // to the previous one if it has no parentMessageId.
  for (let i = 1; i < withIds.length; i++) {
    if (!withIds[i].parentMessageId) {
      withIds[i] = { ...withIds[i], parentMessageId: withIds[i - 1]._stableId };
    }
  }

  // Build children-by-parent map.
  const childrenOf = new Map<string, typeof withIds>();
  for (const msg of withIds) {
    const pid = msg.parentMessageId;
    if (pid) {
      const siblings = childrenOf.get(pid);
      if (siblings) {
        siblings.push(msg);
      } else {
        childrenOf.set(pid, [msg]);
      }
    }
  }

  // Walk the tree from the root.
  const entries: DisplayEntry[] = [];
  const root = withIds[0];
  if (!root) return entries;

  // Helper: process a single message node and continue walking.
  const walk = (node: (typeof withIds)[number]) => {
    if (node.role === "tool") {
      // skip tool messages, proceed to children
    } else if (node.role === "assistant") {
      // Assistant nodes should have been handled as siblings from parent —
      // this path is hit only for root-level assistant messages.
      entries.push({
        type: "branches",
        branches: [{ message: node, flatIndex: node._flatIndex }],
        parentId: node.parentMessageId ?? node._stableId,
      });
    } else {
      // User / system / data messages
      entries.push({ type: "single", message: node, flatIndex: node._flatIndex });
    }

    // Find children of this node.
    const children = childrenOf.get(node._stableId) ?? [];
    if (children.length === 0) return;

    // Separate assistant children (branches) from non-assistant children.
    const assistantChildren = children.filter((c) => c.role === "assistant");
    const otherChildren = children.filter((c) => c.role !== "assistant");

    if (assistantChildren.length > 0) {
      // Sort by branchIndex, then by flat position.
      assistantChildren.sort(
        (a, b) => (a.branchIndex ?? 0) - (b.branchIndex ?? 0) || a._flatIndex - b._flatIndex
      );

      const parentId = node._stableId;
      entries.push({
        type: "branches",
        branches: assistantChildren.map((c) => ({
          message: c,
          flatIndex: c._flatIndex,
        })),
        parentId,
      });

      // Continue walking from the selected branch.
      const selectedIdx = branchSelections.get(parentId) ?? assistantChildren.length - 1;
      const clampedIdx = Math.min(Math.max(0, selectedIdx), assistantChildren.length - 1);
      const selected = assistantChildren[clampedIdx];
      if (selected) {
        // Walk the selected assistant's children (the continuation of that branch).
        const nextChildren = childrenOf.get(selected._stableId) ?? [];
        for (const child of nextChildren) {
          walk(child);
        }
      }
    }

    // Non-assistant children continue linearly.
    for (const child of otherChildren) {
      walk(child);
    }
  };

  walk(root);
  return entries;
};

interface MessageItemProps {
  message: ConversationMessage;
  index: number;
}

const MessageItem = memo(({ message, index }: MessageItemProps) => {
  if (message.role === "tool" || message.role === "assistant") return null;

  const from = message.role === "user" ? "user" : "system";
  const trimmedContent = message.content.trim();
  const hasAttachments = Boolean(message.attachments?.length);

  return (
    <Message from={from}>
      {hasAttachments ? (
        <MessageAttachments
          align="end"
          attachments={message.attachments ?? []}
          messageIndex={index}
        />
      ) : null}
      {trimmedContent ? (
        <>
          <MessageContent>
            <MessageResponse>{message.content}</MessageResponse>
          </MessageContent>
          <MessageActions className="justify-end">
            <CopyButton text={trimmedContent} />
          </MessageActions>
        </>
      ) : null}
    </Message>
  );
});

MessageItem.displayName = "MessageItem";

const AssistantBranchBody = ({ message }: { message: ConversationMessage }) => {
  const metaText = message.modelName ?? null;
  const trimmedContent = message.content.trim();
  const hasCitations = Boolean(message.citations?.length);
  const reasoningText = message.reasoning?.trim() ?? "";
  const hasReasoning = Boolean(reasoningText);
  // Keep the shimmer placeholder only while we have neither visible text nor
  // streamed reasoning — once either arrives, the model-specific UI takes over.
  const showPendingShimmer =
    Boolean(message.reasoningStreaming) && !hasReasoning && !trimmedContent;

  return (
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
        {hasReasoning ? (
          <Reasoning isStreaming={Boolean(message.reasoningStreaming)}>
            <ReasoningTrigger />
            <ReasoningContent>{reasoningText}</ReasoningContent>
          </Reasoning>
        ) : null}
        {showPendingShimmer ? (
          <div className="text-muted-foreground mb-3 flex items-center gap-2 text-sm">
            <span className="bg-current size-2 animate-pulse rounded-full" />
            <Shimmer duration={1.2}>
              {message.reasoningLabel ?? "Processing..."}
            </Shimmer>
          </div>
        ) : null}
        {message.activeToolCall ? (
          <div className="text-muted-foreground mb-3 flex items-center gap-2 text-sm">
            <span className="bg-current size-2 animate-pulse rounded-full" />
            <Shimmer duration={1.2}>Searching documents…</Shimmer>
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
        <MessageMetrics metrics={message.metrics} />
        {message.traceUrl ? (
          <a
            href={message.traceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground mt-1 inline-flex items-center gap-1 text-xs transition-colors"
          >
            <ExternalLinkIcon className="size-3" />
            View trace
          </a>
        ) : null}
      </MessageContent>
    </div>
  );
};

/**
 * Side-by-side comparison grid for multi-model responses to the same prompt.
 * Used when sibling assistant branches have distinct model names.
 */
const ComparisonGridItem = memo(
  ({
    branches,
    parentId,
    selectedBranchIdx,
    onSelectWinner,
    onRegenerateFrom,
  }: {
    branches: BranchEntry[];
    parentId: string;
    selectedBranchIdx: number;
    onSelectWinner: (parentId: string, branchIdx: number, modelId: string) => void;
    onRegenerateFrom: (
      messageId: string,
      modelId: string,
      providerCode: BackendProviderCode,
      replace?: boolean
    ) => Promise<void>;
  }) => {
    const columnClass =
      branches.length === 1
        ? "grid-cols-1"
        : branches.length === 2
          ? "md:grid-cols-2 grid-cols-1"
          : "lg:grid-cols-3 md:grid-cols-2 grid-cols-1";

    return (
      <div className={`grid gap-3 ${columnClass}`}>
        {branches.map(({ message }, branchIdx) => {
          const trimmedContent = message.content.trim();
          const isStreaming = message.reasoningStreaming;
          const isSelected = branchIdx === selectedBranchIdx;
          return (
            <div
              key={message.id ?? `cmp-${message.modelName}`}
              className={cn(
                "bg-card border-border flex flex-col gap-2 rounded-lg border p-3 transition-shadow",
                isSelected && "ring-2 ring-primary"
              )}
            >
              <AssistantBranchBody message={message} />
              {!isStreaming && trimmedContent ? (
                <MessageActions className="justify-end">
                  <CopyButton text={trimmedContent} />
                  <ExpandMessageButton message={message} />
                  <MessageAction
                    tooltip={isSelected ? "Selected" : "Use this response"}
                    onClick={() => onSelectWinner(parentId, branchIdx, message.modelName ?? "")}
                    className={isSelected ? "text-primary" : undefined}
                  >
                    <CheckCircle2Icon className="size-3.5" />
                  </MessageAction>
                  {message.id && message.modelName && message.providerCode ? (
                    <MessageAction
                      tooltip="Regenerate"
                      onClick={() => {
                        void onRegenerateFrom(
                          message.id!,
                          message.modelName!,
                          message.providerCode!,
                          true
                        );
                      }}
                    >
                      <RefreshCwIcon className="size-3.5" />
                    </MessageAction>
                  ) : null}
                </MessageActions>
              ) : null}
            </div>
          );
        })}
      </div>
    );
  }
);

ComparisonGridItem.displayName = "ComparisonGridItem";

/**
 * A sibling group is a "comparison" if its branches have at least 2 distinct
 * model names — i.e. the user asked multiple models the same prompt rather
 * than regenerating with the same model.
 */
const isComparisonGroup = (branches: BranchEntry[]): boolean => {
  if (branches.length < 2) return false;
  const names = new Set(
    branches
      .map((b) => b.message.modelName?.trim())
      .filter((n): n is string => Boolean(n))
  );
  return names.size >= 2;
};

interface BranchedAssistantItemProps {
  branches: BranchEntry[];
  parentId: string;
  selectedBranch: number;
  onBranchChange: (parentId: string, branchIndex: number) => void;
  onRegenerateFrom: (
    messageId: string,
    modelId: string,
    providerCode: BackendProviderCode,
    replace?: boolean
  ) => Promise<void>;
}

const BranchedAssistantItem = memo(({
  branches,
  parentId,
  selectedBranch,
  onBranchChange,
  onRegenerateFrom,
}: BranchedAssistantItemProps) => {
  const clampedIdx = Math.min(Math.max(0, selectedBranch), branches.length - 1);
  const active = branches[clampedIdx]?.message ?? branches[branches.length - 1]?.message;
  const trimmedContent = active?.content.trim() ?? "";
  const isStreaming = active?.reasoningStreaming;

  const handleBranchChange = useCallback(
    (idx: number) => onBranchChange(parentId, idx),
    [onBranchChange, parentId]
  );

  return (
    <Message from="assistant">
      <MessageBranch
        branch={clampedIdx}
        onBranchChange={handleBranchChange}
      >
        <MessageBranchContent>
          {branches.map(({ message }, branchIdx) => (
            <AssistantBranchBody
              key={message.id ?? `branch-${branchIdx}`}
              message={message}
            />
          ))}
        </MessageBranchContent>
        {!isStreaming && trimmedContent ? (
          <MessageToolbar>
            <MessageBranchSelector>
              <MessageBranchPrevious />
              <MessageBranchPage />
              <MessageBranchNext />
            </MessageBranchSelector>
            <MessageActions>
              <CopyButton text={trimmedContent} />
              {active ? <ExpandMessageButton message={active} /> : null}
              {active?.id && active.modelName && active.providerCode ? (
                <MessageAction
                  tooltip="Regenerate"
                  onClick={() => {
                    void onRegenerateFrom(
                      active.id!,
                      active.modelName!,
                      active.providerCode!
                    );
                  }}
                >
                  <RefreshCwIcon className="size-3.5" />
                </MessageAction>
              ) : null}
            </MessageActions>
          </MessageToolbar>
        ) : null}
      </MessageBranch>
    </Message>
  );
});

BranchedAssistantItem.displayName = "BranchedAssistantItem";


let localIdCounter = 0;
const nextLocalId = () => `local-${Date.now()}-${++localIdCounter}`;

const ChatSession = ({ initialMessages, initialThreadId, initialSystemPrompt }: ChatSessionProps) => {
  const router = useRouter();
  const preferredModelId = useMemo(
    () => getLastUsedModelId(initialMessages),
    [initialMessages]
  );
  const [messages, setMessages] = useState<ConversationMessage[]>(() => initialMessages ?? []);
  const [threadId, setThreadId] = useState<string | null>(() => initialThreadId ?? null);
  const [hasThreadRoute, setHasThreadRoute] = useState<boolean>(() => Boolean(initialThreadId));
  const [branchSelections, setBranchSelections] = useState<Map<string, number>>(() => new Map());
  const [winnerModelId, setWinnerModelId] = useState<string | null>(null);

  // Pending config for new chats (kept in memory until first message is sent)
  const [pendingSystemPrompt, setPendingSystemPrompt] = useState<string | null>(
    initialSystemPrompt ?? null
  );
  const [setupDialogOpen, setSetupDialogOpen] = useState(false);
  const [setupSystemPromptDraft, setSetupSystemPromptDraft] = useState("");
  // Whether the first message has been sent (so we stop attaching the pending config)
  const firstMessageSentRef = useRef(false);
  const activeAbortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (initialMessages) {
      setMessages(initialMessages);
    }
    setThreadId(initialThreadId ?? null);
    setHasThreadRoute(Boolean(initialThreadId));
    setBranchSelections(new Map());
  }, [initialMessages, initialThreadId]);

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

  // Stream a single /api/chat call into the assistant message at the given index.
  // Returns the resolved user_message_id and thread_id from the response so that
  // compare-mode follow-up calls can branch under the same user message.
  const streamChatRequest = useCallback(
    async (
      assistantMessageIndex: number,
      requestBody: Record<string, unknown>
    ): Promise<{ userMessageId: string | null; threadId: string | null }> => {
      const abortController = new AbortController();
      activeAbortControllerRef.current = abortController;
      let assistantContentAccumulated = "";
      try {
        const response = await fetch("/api/chat", {
          body: JSON.stringify(requestBody),
          headers: { "Content-Type": "application/json" },
          method: "POST",
          signal: abortController.signal,
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(
            errorText || `Request failed with status ${response.status}`
          );
        }
        if (!response.body) {
          throw new Error("No response stream received.");
        }

        const resolvedThreadId = response.headers.get("x-thread-id");
        const resolvedUserMessageId = response.headers.get("x-user-message-id");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let reasoningContent = "";
        let hasReceivedFirstChunk = false;
        let activeToolCallSet = false;
        let reasoningStreaming = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const raw = decoder.decode(value, { stream: true });
          const { text, citations, messageId, metrics, toolCalls, traceUrl, reasoningDeltas } =
            splitInlineCitations(raw);
          assistantContentAccumulated += text;

          if (citations.length > 0) {
            updateAssistantMessageMeta(assistantMessageIndex, { citations });
          }
          if (messageId) {
            updateAssistantMessageMeta(assistantMessageIndex, { id: messageId });
          }
          if (metrics) {
            updateAssistantMessageMeta(assistantMessageIndex, { metrics });
          }
          if (traceUrl) {
            updateAssistantMessageMeta(assistantMessageIndex, { traceUrl });
          }
          if (toolCalls.length > 0) {
            activeToolCallSet = true;
            updateAssistantMessageMeta(assistantMessageIndex, {
              activeToolCall: toolCalls[toolCalls.length - 1],
            });
          }
          if (reasoningDeltas.length > 0) {
            reasoningContent += reasoningDeltas.join("");
            reasoningStreaming = true;
            updateAssistantMessageMeta(assistantMessageIndex, {
              reasoning: reasoningContent,
              reasoningStreaming: true,
            });
          }
          if (!hasReceivedFirstChunk) {
            hasReceivedFirstChunk = true;
            // Keep the placeholder shimmer if we are only seeing reasoning so
            // far; drop it as soon as any visible text arrives.
            if (text) {
              updateAssistantMessageMeta(assistantMessageIndex, {
                reasoningStreaming: false,
              });
            }
          }
          if (text) {
            if (reasoningStreaming) {
              reasoningStreaming = false;
              updateAssistantMessageMeta(assistantMessageIndex, {
                reasoningStreaming: false,
              });
            }
            if (activeToolCallSet) {
              activeToolCallSet = false;
              updateAssistantMessageMeta(assistantMessageIndex, { activeToolCall: null });
            }
            updateAssistantMessage(assistantMessageIndex, assistantContentAccumulated);
          }
        }

        assistantContentAccumulated += decoder.decode();
        updateAssistantMessageMeta(assistantMessageIndex, {
          reasoningStreaming: false,
        });
        updateAssistantMessage(
          assistantMessageIndex,
          assistantContentAccumulated.trim() ? assistantContentAccumulated : "(No response)"
        );

        return {
          userMessageId: resolvedUserMessageId,
          threadId: resolvedThreadId,
        };
      } catch (error) {
        updateAssistantMessageMeta(assistantMessageIndex, { reasoningStreaming: false });
        if (error instanceof Error && error.name === "AbortError") {
          const partial = assistantContentAccumulated.trim();
          updateAssistantMessage(
            assistantMessageIndex,
            partial ? `${partial}\n\n*(Generation stopped)*` : "*(Generation stopped)*"
          );
          return { userMessageId: null, threadId: null };
        }
        const errorMessage =
          error instanceof Error ? error.message : "Unexpected chat error.";
        updateAssistantMessage(assistantMessageIndex, `Error: ${errorMessage}`);
        return { userMessageId: null, threadId: null };
      } finally {
        if (activeAbortControllerRef.current === abortController) {
          activeAbortControllerRef.current = null;
        }
      }
    },
    [updateAssistantMessage, updateAssistantMessageMeta]
  );

  const handleSubmitMessage = useCallback(async (payload: {
    message: PromptInputMessage;
    modelId: string;
    providerCode: BackendProviderCode;
    regenerateFromMessageId?: string;
    replaceExisting?: boolean;
    compareWith?: CompareModelSelection[];
  }) => {
    const userContent = normalizeMessageText(payload.message);
    const userAttachments = payload.message.files ?? [];
    const attachmentCount = userAttachments.length;
    const reasoningLabel = buildPendingReasoningLabel(attachmentCount);

    if (!userContent && userAttachments.length === 0 && !payload.regenerateFromMessageId) {
      return;
    }

    // Determine which assistant message the new user message continues from
    // by finding the tail of the currently visible branch path.
    let continueFromMessageId: string | undefined;
    const compareModels = payload.compareWith ?? [];
    const isCompareMode = compareModels.length > 0;

    let leadAssistantIndex = -1;
    let compareAssistantIndices: number[] = [];

    setMessages((prev) => {
      if (payload.regenerateFromMessageId) {
        // Regenerating: the new assistant message is a sibling of the one
        // being regenerated, so the parent is the same parent as that message.
        const regenMsg = prev.find((m) => m.id === payload.regenerateFromMessageId);
        const parentId = regenMsg?.parentMessageId ?? undefined;
        // In replace mode (comparison), remove the old message instead of branching.
        const base = payload.replaceExisting
          ? prev.filter((m) => m.id !== payload.regenerateFromMessageId)
          : prev;
        leadAssistantIndex = base.length;
        return [
          ...base,
          {
            role: "assistant" as const,
            content: "",
            providerCode: payload.providerCode,
            modelName: payload.modelId,
            reasoningLabel,
            reasoningStreaming: true,
            parentMessageId: parentId ?? null,
          },
        ];
      }

      // Use the last visible assistant from the tree-walk display.
      continueFromMessageId = lastVisibleAssistantIdRef.current;

      const userLocalId = nextLocalId();
      const userMessage: ConversationMessage = {
        id: userLocalId,
        role: "user",
        content: userContent,
        attachments: userAttachments,
        parentMessageId: continueFromMessageId ?? null,
      };

      const placeholders: ConversationMessage[] = [
        {
          role: "assistant" as const,
          content: "",
          providerCode: payload.providerCode,
          modelName: payload.modelId,
          reasoningLabel,
          reasoningStreaming: true,
          parentMessageId: userLocalId,
        },
        ...compareModels.map((entry) => ({
          role: "assistant" as const,
          content: "",
          providerCode: entry.providerCode,
          modelName: entry.modelId,
          reasoningLabel,
          reasoningStreaming: true,
          parentMessageId: userLocalId,
        })),
      ];

      leadAssistantIndex = prev.length + 1;
      compareAssistantIndices = compareModels.map(
        (_entry, idx) => leadAssistantIndex + 1 + idx
      );
      return [...prev, userMessage, ...placeholders];
    });

    const shouldRedirectToThread = !hasThreadRoute;

    // Attach pending config only on the first message of a new chat
    const isFirstMessage = !firstMessageSentRef.current && !hasThreadRoute;
    const firstMessageExtras: Record<string, string> = {};
    if (isFirstMessage) {
      if (pendingSystemPrompt) firstMessageExtras.systemPrompt = pendingSystemPrompt;
      firstMessageSentRef.current = true;
    }

    const baseRequestBody = {
      attachments: userAttachments.map(toBackendChatAttachment),
      prompt: userContent,
      threadId,
      modelId: payload.modelId,
      providerCode: payload.providerCode,
      ...(payload.regenerateFromMessageId
        ? { regenerateFromMessageId: payload.regenerateFromMessageId }
        : {}),
      ...(continueFromMessageId ? { continueFromMessageId } : {}),
      ...firstMessageExtras,
    };

    // Lead request: creates the user message (or branches off the existing one
    // when regenerating). Compare-mode siblings are fired only after this
    // resolves so they can target the new user message via compareWithUserMessageId.
    const leadResult = await streamChatRequest(leadAssistantIndex, baseRequestBody);

    if (leadResult.threadId && leadResult.threadId !== threadId) {
      setThreadId(leadResult.threadId);
    }
    if (leadResult.threadId && shouldRedirectToThread) {
      window.history.replaceState(
        window.history.state,
        "",
        `/chats/${leadResult.threadId}`
      );
      setHasThreadRoute(true);
    }

    if (isCompareMode && leadResult.userMessageId) {
      // Fire all sibling-model requests in parallel.
      await Promise.all(
        compareModels.map((entry, idx) =>
          streamChatRequest(compareAssistantIndices[idx], {
            attachments: [],
            prompt: "",
            threadId: leadResult.threadId ?? threadId,
            modelId: entry.modelId,
            providerCode: entry.providerCode,
            compareWithUserMessageId: leadResult.userMessageId,
          })
        )
      );
    }

    if (shouldRedirectToThread && leadResult.threadId) {
      router.replace(`/chats/${leadResult.threadId}`);
    }
    window.dispatchEvent(new Event("chat-threads-updated"));
  }, [hasThreadRoute, pendingSystemPrompt, router, streamChatRequest, threadId]);

  const handleRegenerateFrom = useCallback(
    async (
      messageId: string,
      modelId: string,
      providerCode: BackendProviderCode,
      replace = false
    ) => {
      if (!threadId) return;

      // Find the parent of the message being regenerated so we can auto-select
      // the new branch (which will be appended last).
      const regenMsg = messages.find((m) => m.id === messageId);
      const parentId = regenMsg?.parentMessageId;

      await handleSubmitMessage({
        message: { text: "", files: [] },
        modelId,
        providerCode,
        regenerateFromMessageId: messageId,
        replaceExisting: replace,
      });

      // In replace mode, the regen is done — now hard-delete the old message
      // from the backend so it doesn't reappear on refresh.
      if (replace && threadId) {
        try {
          await fetch(`/api/threads/${threadId}/messages/${messageId}`, { method: "DELETE" });
        } catch {
          // Non-critical: local state is already correct; ignore network errors.
        }
      }

      // After regeneration in branch mode, select the newest branch (appended last).
      // In replace mode the message count doesn't change so no selection update needed.
      if (!replace && parentId) {
        setBranchSelections((prev) => {
          const next = new Map(prev);
          // Count how many assistant siblings exist for this parent after the
          // new one is appended. We set a high number — buildTreeDisplay will
          // clamp it to the last valid index.
          next.set(parentId, Number.MAX_SAFE_INTEGER);
          return next;
        });
      }
    },
    [threadId, handleSubmitMessage, messages]
  );

  const handleBranchChange = useCallback(
    (parentId: string, branchIndex: number) => {
      setBranchSelections((prev) => {
        const next = new Map(prev);
        next.set(parentId, branchIndex);
        return next;
      });
    },
    []
  );

  const handleSelectWinner = useCallback(
    (parentId: string, branchIndex: number, modelId: string) => {
      handleBranchChange(parentId, branchIndex);
      if (modelId) {
        setWinnerModelId(modelId);
      }
    },
    [handleBranchChange]
  );

  const displayEntries = useMemo(
    () => buildTreeDisplay(messages, branchSelections),
    [messages, branchSelections]
  );

  // Track the last visible assistant message ID so handleSubmitMessage can
  // determine the correct continueFromMessageId without stale closures.
  const lastVisibleAssistantIdRef = useRef<string | undefined>(undefined);
  useEffect(() => {
    for (let i = displayEntries.length - 1; i >= 0; i--) {
      const entry = displayEntries[i];
      if (entry.type === "branches") {
        const sel = branchSelections.get(entry.parentId) ?? entry.branches.length - 1;
        const clamped = Math.min(Math.max(0, sel), entry.branches.length - 1);
        const msg = entry.branches[clamped]?.message;
        if (msg?.id) {
          lastVisibleAssistantIdRef.current = msg.id;
          return;
        }
      }
    }
    lastVisibleAssistantIdRef.current = undefined;
  }, [displayEntries, branchSelections]);

  const { register, unregister } = useChatComposer();

  const handleStop = useCallback(() => {
    activeAbortControllerRef.current?.abort();
  }, []);

  useEffect(() => {
    register({ onSubmitMessage: handleSubmitMessage, onStop: handleStop, preferredModelId, forceModelId: winnerModelId });
    return () => unregister();
  }, [register, unregister, handleSubmitMessage, handleStop, preferredModelId, winnerModelId]);

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden">
      <SystemPromptDialog
        open={setupDialogOpen}
        value={setupSystemPromptDraft}
        isSubmitting={false}
        onValueChange={setSetupSystemPromptDraft}
        onSubmit={() => {
          setPendingSystemPrompt(setupSystemPromptDraft.trim() || null);
          setSetupDialogOpen(false);
        }}
        onClose={() => setSetupDialogOpen(false)}
      />
      <Conversation className="min-h-0 min-w-0 flex-1">
        <ConversationContent
          className={`mx-auto w-full max-w-3xl ${
            messages.length === 0 ? "h-full justify-center" : "min-h-full"
          }`}
        >
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-4">
              <ConversationEmptyState
                className="[&_h3]:text-2xl [&_h3]:font-semibold [&_p]:text-base"
                description="Ask me anything to get started. I am here to help."
                title="Ready when you are"
              />
              {!initialThreadId ? (
                <div className="flex flex-col items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="gap-1.5"
                    onClick={() => {
                      setSetupSystemPromptDraft(pendingSystemPrompt ?? "");
                      setSetupDialogOpen(true);
                    }}
                  >
                    <Settings2Icon className="size-3.5" />
                    Configure prompt
                  </Button>
                  {pendingSystemPrompt ? (
                    <div className="flex flex-wrap justify-center gap-1.5">
                      <Badge variant="secondary" className="gap-1 text-xs">
                        <BotIcon className="size-3" />
                        System prompt set
                      </Badge>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : (
            displayEntries.map((entry) =>
              entry.type === "single" ? (
                <MessageItem
                  key={`single-${entry.flatIndex}-${entry.message.role}`}
                  index={entry.flatIndex}
                  message={entry.message}
                />
              ) : isComparisonGroup(entry.branches) ? (
                <ComparisonGridItem
                  key={`compare-${entry.parentId}`}
                  branches={entry.branches}
                  parentId={entry.parentId}
                  selectedBranchIdx={branchSelections.get(entry.parentId) ?? entry.branches.length - 1}
                  onSelectWinner={handleSelectWinner}
                  onRegenerateFrom={handleRegenerateFrom}
                />
              ) : (
                <BranchedAssistantItem
                  key={`branches-${entry.parentId}`}
                  branches={entry.branches}
                  parentId={entry.parentId}
                  selectedBranch={branchSelections.get(entry.parentId) ?? entry.branches.length - 1}
                  onBranchChange={handleBranchChange}
                  onRegenerateFrom={handleRegenerateFrom}
                />
              )
            )
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>
    </div>
  );
};

export default ChatSession;
