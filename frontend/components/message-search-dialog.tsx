"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { MessageSquareIcon } from "lucide-react"

type SearchResult = {
  message_id: string
  thread_id: string
  thread_title: string | null
  role: "user" | "assistant"
  content_snippet: string
  created_at: string
}

type MessageSearchDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function MessageSearchDialog({ open, onOpenChange }: MessageSearchDialogProps) {
  const router = useRouter()
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const search = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResults([])
      return
    }
    setLoading(true)
    try {
      const response = await fetch(`/api/messages/search?q=${encodeURIComponent(q)}`)
      if (response.ok) {
        setResults((await response.json()) as SearchResult[])
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => void search(query), 300)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [query, search])

  useEffect(() => {
    if (!open) {
      setQuery("")
      setResults([])
    }
  }, [open])

  const handleSelect = (threadId: string) => {
    onOpenChange(false)
    router.push(`/chats/${threadId}`)
  }

  const isEmpty = query.trim() === ""
  const heading = results.length === 1 ? "1 result" : `${results.length} results`

  return (
    <CommandDialog
      open={open}
      onOpenChange={onOpenChange}
      title="Search messages"
      description="Search across all your chat messages"
    >
      <Command shouldFilter={false}>
        <CommandInput
          placeholder="Search messages..."
          value={query}
          onValueChange={setQuery}
        />
        <CommandList>
          {loading ? (
            <CommandEmpty>Searching...</CommandEmpty>
          ) : isEmpty ? (
            <CommandEmpty>Type to search across all your chats.</CommandEmpty>
          ) : results.length === 0 ? (
            <CommandEmpty>No messages found for &ldquo;{query}&rdquo;.</CommandEmpty>
          ) : (
            <CommandGroup heading={heading}>
              {results.map((result) => (
                <CommandItem
                  key={result.message_id}
                  value={result.message_id}
                  onSelect={() => handleSelect(result.thread_id)}
                  className="items-start"
                >
                  <MessageSquareIcon className="mt-0.5 shrink-0 text-muted-foreground" />
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <span className="text-xs font-medium text-muted-foreground">
                      {result.thread_title ?? "Untitled"}
                      {" · "}
                      {result.role === "user" ? "You" : "Assistant"}
                    </span>
                    <span className="truncate text-sm">{result.content_snippet}</span>
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          )}
        </CommandList>
      </Command>
    </CommandDialog>
  )
}
