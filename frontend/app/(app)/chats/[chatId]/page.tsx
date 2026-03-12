import { notFound } from "next/navigation";

import ChatSession from "@/components/chat-session";
import type { ConversationMessage } from "@/components/ai-elements/conversation";
import { fetchBackend, getServerSessionToken } from "@/lib/backend";

type ChatPageProps = {
  params: Promise<{
    chatId: string;
  }>;
};

type ThreadMessageResponse = {
  role: string;
  content: string;
  provider_code?: string | null;
  model_name?: string | null;
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
    providerCode === "xai" ||
    providerCode === "openrouter" ||
    providerCode === "other"
  ) {
    return providerCode;
  }
  return undefined;
};

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
  const initialMessages: ConversationMessage[] = thread.messages.map((message) => ({
    role: toConversationRole(message.role),
    content: message.content,
    providerCode: toProviderCode(message.provider_code),
    modelName: message.model_name ?? null,
  }));

  return <ChatSession initialMessages={initialMessages} initialThreadId={thread.id} />;
}
