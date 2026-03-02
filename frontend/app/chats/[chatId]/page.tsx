import ChatSession from "@/components/chat-session";
import { getChatThreadById } from "@/lib/chat-threads";
import { notFound } from "next/navigation";

type ChatPageProps = {
    params: Promise<{
        chatId: string;
    }>;
};

export default async function ChatPage({ params }: ChatPageProps) {
    const { chatId } = await params;
    const chat = getChatThreadById(chatId);

    if (!chat) {
        notFound();
    }

    return <ChatSession initialMessages={chat.messages} />;
}
