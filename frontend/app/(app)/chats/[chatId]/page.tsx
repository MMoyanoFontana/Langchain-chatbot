import { notFound } from "next/navigation";

import ChatSession from "@/components/chat-session";
import type { ConversationMessage } from "@/components/ai-elements/conversation";
import {
  toConversationCitation,
  toConversationMetrics,
  type BackendChatCitation,
} from "@/lib/chat-citations";
import { toChatAttachment, type BackendChatAttachment } from "@/lib/chat-attachments";
import { fetchBackend, getServerSessionToken } from "@/lib/backend";

type ChatPageProps = {
  params: Promise<{
    chatId: string;
  }>;
};

type ThreadMessageResponse = {
  id?: string;
  role: string;
  content: string;
  reasoning_content?: string | null;
  attachments?: BackendChatAttachment[] | null;
  citations?: BackendChatCitation[] | null;
  provider_code?: string | null;
  model_name?: string | null;
  parent_message_id?: string | null;
  branch_index?: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  latency_ms?: number | null;
  time_to_first_token_ms?: number | null;
};

type ThreadResponse = {
  id: string;
  messages: ThreadMessageResponse[];
};

const toConversationRole = (role: string): ConversationMessage["role"] => {
  if (
    role === "user" ||
    role === "assistant" ||
    role === "system" ||
    role === "data" ||
    role === "tool"
  ) {
    return role;
  }
  return "assistant";
};

const toProviderCode = (
  providerCode: string | null | undefined
): ConversationMessage["providerCode"] => {
  if (
    providerCode === "openai" ||
    providerCode === "anthropic" ||
    providerCode === "gemini" ||
    providerCode === "groq" ||
    providerCode === "other"
  ) {
    return providerCode;
  }
  return undefined;
};

const isDefined = <Value,>(value: Value | null): value is Value => value !== null;

export default async function ChatPage({ params }: ChatPageProps) {
  const sessionToken = await getServerSessionToken();
  if (!sessionToken) {
    notFound();
  }

  const { chatId } = await params;
  const response = await fetchBackend(
    `/users/me/threads/${chatId}`,
    {
      headers: { Accept: "application/json" },
    },
    sessionToken
  );

  if (!response.ok) {
    notFound();
  }

  const thread = (await response.json()) as ThreadResponse;
  const initialMessages: ConversationMessage[] = thread.messages
    .filter((message) => message.role !== "tool")
    .map((message) => ({
      id: message.id,
      attachments:
        message.attachments?.map(toChatAttachment).filter(isDefined) || undefined,
      citations:
        message.citations?.map(toConversationCitation).filter(isDefined) || undefined,
      role: toConversationRole(message.role),
      content: message.content,
      reasoning: message.reasoning_content ?? null,
      providerCode: toProviderCode(message.provider_code),
      modelName: message.model_name ?? null,
      parentMessageId: message.parent_message_id ?? null,
      branchIndex: message.branch_index ?? 0,
      metrics: toConversationMetrics({
        prompt_tokens: message.prompt_tokens,
        completion_tokens: message.completion_tokens,
        total_tokens: message.total_tokens,
        latency_ms: message.latency_ms,
        time_to_first_token_ms: message.time_to_first_token_ms,
      }),
    }));
  return (
    <ChatSession
      initialMessages={initialMessages}
      initialThreadId={thread.id}
    />
  );
}
