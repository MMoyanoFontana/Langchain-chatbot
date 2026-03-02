export type ChatMessageRole = "user" | "assistant" | "system" | "data" | "tool";

export interface ChatMessage {
  role: ChatMessageRole;
  content: string;
}

export interface ChatSummary {
  id: string;
  title: string;
  timestamp: string;
}

export interface ChatThread extends ChatSummary {
  messages: ChatMessage[];
}

export const CHAT_THREADS: ChatThread[] = [
  {
    id: "chat-1",
    title: "Planning sprint goals",
    timestamp: "2m ago",
    messages: [
      {
        role: "user",
        content: "Help me draft sprint goals for a 2-week cycle focused on reliability.",
      },
      {
        role: "assistant",
        content:
          "Use three measurable goals: reduce API error rate, improve test pass rate, and cut on-call pages. I can suggest concrete target numbers if you share your baseline.",
      },
    ],
  },
  {
    id: "chat-2",
    title: "Fixing sidebar keyboard shortcuts",
    timestamp: "14m ago",
    messages: [
      {
        role: "user",
        content: "Ctrl+Shift+, should open settings, but it sometimes does nothing.",
      },
      {
        role: "assistant",
        content:
          "Check focus traps and whether the event listener is attached at `window`. Also guard against keymap conflicts in the browser.",
      },
    ],
  },
  {
    id: "chat-3",
    title: "Tailwind spacing audit",
    timestamp: "1h ago",
    messages: [
      {
        role: "user",
        content: "Can you review spacing consistency in cards and sidebars?",
      },
      {
        role: "assistant",
        content:
          "Start by standardizing container padding tokens (`p-2`, `p-3`, `p-4`) and mapping each component family to one token set.",
      },
    ],
  },
  {
    id: "chat-4",
    title: "How to deploy LangChain agent",
    timestamp: "Yesterday",
    messages: [
      {
        role: "user",
        content: "What is the quickest deployment path for a LangChain chatbot?",
      },
      {
        role: "assistant",
        content:
          "Use a stateless API deployment first, store chat transcripts externally, and add streaming responses with a single `/api/chat` endpoint.",
      },
    ],
  },
  {
    id: "chat-5",
    title: "RAG chunking strategy notes",
    timestamp: "2 days ago",
    messages: [
      {
        role: "user",
        content: "What chunk size should I use for technical documentation in RAG?",
      },
      {
        role: "assistant",
        content:
          "A practical starting point is 400-800 tokens with 10-20% overlap. Adjust by retrieval precision and answer completeness.",
      },
    ],
  },
];

export const CHAT_HISTORY: ChatSummary[] = CHAT_THREADS.map(
  ({ id, title, timestamp }) => ({
    id,
    title,
    timestamp,
  })
);

export const getChatThreadById = (chatId: string): ChatThread | undefined =>
  CHAT_THREADS.find((chat) => chat.id === chatId);
