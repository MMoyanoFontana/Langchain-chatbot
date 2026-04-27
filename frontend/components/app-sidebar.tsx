"use client"

import { useState } from "react"
import { usePathname, useRouter } from "next/navigation"
import Link from "next/link"
import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuAction,
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar"
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,

} from "@/components/ui/collapsible"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { DeleteThreadDialog } from "@/components/delete-thread-dialog"
import { MessageSearchDialog } from "@/components/message-search-dialog"
import { NavUser } from "@/components/nav-user"
import { RenameThreadDialog } from "@/components/rename-thread-dialog"
import { SettingsDialog } from "@/components/settings-dialog"
import { SystemPromptDialog } from "@/components/system-prompt-dialog"
import { useKeyboardShortcut } from "@/hooks/use-keyboard-shortcut"
import type { SidebarUser } from "@/hooks/use-sidebar-user"
import { useSidebarUser } from "@/hooks/use-sidebar-user"
import { MAX_THREAD_TITLE_LENGTH, useThreadActions } from "@/hooks/use-thread-actions"
import { useThreadHistory } from "@/hooks/use-thread-history"
import { BotIcon, ChevronRight, DownloadIcon, FileJsonIcon, FileTextIcon, MoreHorizontal, PencilIcon, PlusCircle, Route, SearchIcon, Trash2 } from "lucide-react"
import {
    DropdownMenuSub,
    DropdownMenuSubContent,
    DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu"

const APP_NAME = "LLM Compare"

async function exportThread(threadId: string, title: string | null, format: "markdown" | "json") {
    const response = await fetch(`/api/threads/${threadId}/export?format=${format}`)
    if (!response.ok) return
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    const safeName = (title ?? threadId).slice(0, 80).replace(/[/\\:*?"<>|]/g, "_")
    a.download = `${safeName}.${format === "json" ? "json" : "md"}`
    a.click()
    URL.revokeObjectURL(url)
}

type AppSidebarProps = {
    initialUser?: SidebarUser | null
}

export function AppSidebar({ initialUser }: AppSidebarProps) {
    const router = useRouter()
    const [settingsOpen, setSettingsOpen] = useState(false)
    const [searchOpen, setSearchOpen] = useState(false)
    const [isLoggingOut, setIsLoggingOut] = useState(false)
    const pathname = usePathname()

    const {
        user: appUser,
        isLoading: userLoading,
        refresh: refreshUser,
    } = useSidebarUser(initialUser)

    const {
        threads: chatHistory,
        isLoading: historyLoading,
        error: historyError,
        refresh: refreshHistory,
    } = useThreadHistory(pathname)

    const {
        threadActionLoadingId,
        renameDialogOpen,
        deleteDialogOpen,
        systemPromptDialogOpen,
        activeThread,
        renameValue,
        renameError,
        deleteError,
        systemPromptValue,
        systemPromptError,
        renameValidationError,
        canSubmitRename,
        canSubmitDelete,
        setRenameValue,
        setSystemPromptValue,
        openRenameDialog,
        openDeleteDialog,
        openSystemPromptDialog,
        closeRenameDialog,
        closeDeleteDialog,
        closeSystemPromptDialog,
        submitRename,
        submitDelete,
        submitSystemPrompt,
    } = useThreadActions({
        pathname,
        refreshHistory,
        navigateToRoot: () => {
            router.replace("/")
        },
    })

    useKeyboardShortcut(["ctrl", "shift", ","], () => {
        setSettingsOpen((previous) => {
            const next = !previous
            if (!next) {
                void refreshUser()
            }
            return next
        })
    })

    const handleSettingsOpenChange = (open: boolean) => {
        setSettingsOpen(open)
        if (!open) {
            void refreshUser()
        }
    }

    const handleLogout = async () => {
        if (isLoggingOut) {
            return
        }

        setIsLoggingOut(true)
        try {
            await fetch("/api/auth/logout", { method: "POST" })
        } finally {
            router.replace("/login")
            router.refresh()
            setIsLoggingOut(false)
        }
    }

    return (
        <>
            <Sidebar collapsible="icon">
                <SidebarHeader>
                    <SidebarMenu>
                        <SidebarMenuItem>
                            <SidebarMenuButton
                                size="lg"
                                tooltip={APP_NAME}
                                className="hover:bg-transparent active:bg-transparent cursor-default"
                            >
                                <span className="flex aspect-square size-8 items-center justify-center rounded-lg border border-sidebar-border/70 bg-sidebar-accent/60 text-sidebar-foreground">
                                    <Route className="size-4" />
                                </span>
                                <span className="truncate text-base font-semibold">
                                    {APP_NAME}
                                </span>
                            </SidebarMenuButton>
                        </SidebarMenuItem>
                    </SidebarMenu>
                </SidebarHeader>
                <SidebarContent>
                    <SidebarGroup key="actions">
                        <SidebarGroupContent>
                            <SidebarMenu>
                                <SidebarMenuItem key="new-chat">
                                    <SidebarMenuButton
                                        render={<Link href="/" />}
                                        onClick={() => void refreshHistory()}
                                    >
                                        <PlusCircle />
                                        <span>New chat</span>
                                    </SidebarMenuButton >
                                </SidebarMenuItem>
                                <SidebarMenuItem key="search-chat">
                                    <SidebarMenuButton onClick={() => setSearchOpen(true)}>
                                        <SearchIcon />
                                        <span>Search</span>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>
                            </SidebarMenu>
                        </SidebarGroupContent>
                    </SidebarGroup>
                    <Collapsible defaultOpen className="group/collapsible">
                        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
                            <SidebarGroupLabel className="transition-colors hover:bg-muted" render={<CollapsibleTrigger className="group hover:cursor-pointer">
                                History
                                <ChevronRight className="ml-auto group-data-[panel-open]:rotate-90" />
                            </CollapsibleTrigger>}>
                            </SidebarGroupLabel>
                            <CollapsibleContent>
                                <SidebarGroupContent>
                                    <SidebarMenu>
                                        {historyLoading ? (
                                            <SidebarMenuItem key="history-loading">
                                                <SidebarMenuButton disabled>
                                                    <span>Loading chats...</span>
                                                </SidebarMenuButton>
                                            </SidebarMenuItem>
                                        ) : historyError ? (
                                            <SidebarMenuItem key="history-error">
                                                <SidebarMenuButton disabled>
                                                    <span>{historyError}</span>
                                                </SidebarMenuButton>
                                            </SidebarMenuItem>
                                        ) : chatHistory.length === 0 ? (
                                            <SidebarMenuItem key="history-empty">
                                                <SidebarMenuButton disabled>
                                                    <span>No chats yet</span>
                                                </SidebarMenuButton>
                                            </SidebarMenuItem>
                                        ) : (
                                            chatHistory.map((chat) => {
                                                const href = `/chats/${chat.id}`
                                                const isActionLoading = threadActionLoadingId === chat.id

                                                return (
                                                    <SidebarMenuItem key={chat.id}>
                                                        <SidebarMenuButton
                                                            isActive={pathname === href}
                                                            render={<Link href={href} />}
                                                        >
                                                            <span>{chat.title}</span>
                                                        </SidebarMenuButton>
                                                        <DropdownMenu>
                                                            <DropdownMenuTrigger
                                                                render={
                                                                    <SidebarMenuAction
                                                                        showOnHover
                                                                        onClick={(event) => {
                                                                            event.preventDefault()
                                                                            event.stopPropagation()
                                                                        }}
                                                                    >
                                                                        <MoreHorizontal />
                                                                        <span className="sr-only">Chat actions</span>
                                                                    </SidebarMenuAction>
                                                                }
                                                            />
                                                            <DropdownMenuContent side="right" align="start">
                                                                <DropdownMenuItem
                                                                    disabled={isActionLoading}
                                                                    onClick={() => {
                                                                        openRenameDialog(chat)
                                                                    }}
                                                                >
                                                                    <PencilIcon />
                                                                    Rename
                                                                </DropdownMenuItem>
                                                                <DropdownMenuItem
                                                                    disabled={isActionLoading}
                                                                    onClick={() => {
                                                                        openSystemPromptDialog(chat)
                                                                    }}
                                                                >
                                                                    <BotIcon />
                                                                    System Prompt
                                                                </DropdownMenuItem>
                                                                <DropdownMenuSub>
                                                                    <DropdownMenuSubTrigger disabled={isActionLoading}>
                                                                        <DownloadIcon />
                                                                        Export
                                                                    </DropdownMenuSubTrigger>
                                                                    <DropdownMenuSubContent>
                                                                        <DropdownMenuItem
                                                                            onClick={() => {
                                                                                void exportThread(chat.id, chat.title ?? null, "markdown")
                                                                            }}
                                                                        >
                                                                            <FileTextIcon />
                                                                            Markdown
                                                                        </DropdownMenuItem>
                                                                        <DropdownMenuItem
                                                                            onClick={() => {
                                                                                void exportThread(chat.id, chat.title ?? null, "json")
                                                                            }}
                                                                        >
                                                                            <FileJsonIcon />
                                                                            JSON
                                                                        </DropdownMenuItem>
                                                                    </DropdownMenuSubContent>
                                                                </DropdownMenuSub>
                                                                <DropdownMenuSeparator />
                                                                <DropdownMenuItem
                                                                    variant="destructive"
                                                                    disabled={isActionLoading}
                                                                    onClick={() => {
                                                                        openDeleteDialog(chat)
                                                                    }}
                                                                >
                                                                    <Trash2 />
                                                                    Delete
                                                                </DropdownMenuItem>
                                                            </DropdownMenuContent>
                                                        </DropdownMenu>
                                                    </SidebarMenuItem>
                                                )
                                            })
                                        )}
                                    </SidebarMenu>
                                </SidebarGroupContent>
                            </CollapsibleContent>
                        </SidebarGroup>
                    </Collapsible>
                </SidebarContent>
                <SidebarFooter>
                    {userLoading ? (
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton disabled>
                                    <span>Loading account...</span>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    ) : appUser ? (
                        <NavUser
                            user={appUser}
                            onSettingsClick={() => handleSettingsOpenChange(true)}
                            onLogout={() => {
                                void handleLogout()
                            }}
                        />
                    ) : (
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton disabled>
                                    <span>No signed-in user</span>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    )}
                </SidebarFooter>
            </Sidebar>
            <SettingsDialog open={settingsOpen} onOpenChange={handleSettingsOpenChange} />
            <MessageSearchDialog open={searchOpen} onOpenChange={setSearchOpen} />
            <RenameThreadDialog
                open={renameDialogOpen}
                value={renameValue}
                validationError={renameValidationError}
                requestError={renameError}
                isSubmitting={Boolean(threadActionLoadingId)}
                canSubmit={canSubmitRename}
                maxLength={MAX_THREAD_TITLE_LENGTH}
                onValueChange={setRenameValue}
                onSubmit={() => {
                    void submitRename()
                }}
                onClose={closeRenameDialog}
            />
            <DeleteThreadDialog
                open={deleteDialogOpen}
                threadTitle={activeThread?.title}
                error={deleteError}
                isSubmitting={Boolean(threadActionLoadingId)}
                canSubmit={canSubmitDelete}
                onSubmit={() => {
                    void submitDelete()
                }}
                onClose={closeDeleteDialog}
            />
            <SystemPromptDialog
                open={systemPromptDialogOpen}
                value={systemPromptValue}
                requestError={systemPromptError}
                isSubmitting={Boolean(threadActionLoadingId)}
                onValueChange={setSystemPromptValue}
                onSubmit={() => {
                    void submitSystemPrompt()
                }}
                onClose={closeSystemPromptDialog}
            />
        </>
    )
}
