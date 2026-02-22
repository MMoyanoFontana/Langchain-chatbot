"use client"

import { useState } from "react"
import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar"
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,

} from "@/components/ui/collapsible"
import { NavUser } from "@/components/nav-user"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog"
import { useKeyboardShortcut } from "@/hooks/use-keyboard-shortcut"
import { ChevronDown, PlusCircle, SearchIcon } from "lucide-react"

export function AppSidebar() {
    const [settingsOpen, setSettingsOpen] = useState(false)
    const chatHistory = [
        { id: "chat-1", title: "Planning sprint goals", timestamp: "2m ago" },
        { id: "chat-2", title: "Fixing sidebar keyboard shortcuts", timestamp: "14m ago" },
        { id: "chat-3", title: "Tailwind spacing audit", timestamp: "1h ago" },
        { id: "chat-4", title: "How to deploy LangChain agent", timestamp: "Yesterday" },
        { id: "chat-5", title: "RAG chunking strategy notes", timestamp: "2 days ago" },
    ]

    const data = {
        user: {
            name: "Test User",
            email: "test@example.com",
            avatar: "/avatars/shadcn.jpg",
        },
    }

    useKeyboardShortcut(["ctrl", "shift", ","], () => setSettingsOpen(!settingsOpen))

    return (
        <>
            <Sidebar collapsible="icon">
                <SidebarHeader className="h-12" />
                <SidebarContent>
                    <SidebarGroup key="actions">
                        <SidebarGroupContent>
                            <SidebarMenu>
                                <SidebarMenuItem key="new-chat">
                                    <SidebarMenuButton>
                                        <PlusCircle />
                                        <span>New chat</span>
                                    </SidebarMenuButton >
                                </SidebarMenuItem>
                                <SidebarMenuItem key="search-chat">
                                    <SidebarMenuButton>
                                        <SearchIcon />
                                        <span>Search</span>
                                    </SidebarMenuButton >
                                </SidebarMenuItem>
                            </SidebarMenu>
                        </SidebarGroupContent>
                    </SidebarGroup>
                    <Collapsible defaultOpen className="group/collapsible">
                        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
                            <SidebarGroupLabel render={<CollapsibleTrigger>
                                History
                                <ChevronDown className="ml-auto transition-transform group-data-[state=open]/collapsible:rotate-180" />
                            </CollapsibleTrigger>}>
                            </SidebarGroupLabel>
                            <CollapsibleContent>
                                <SidebarGroupContent>
                                    <SidebarMenu>
                                        {chatHistory.map((chat) => (
                                            <SidebarMenuItem key={chat.id}>
                                                <SidebarMenuButton render={<a href={`/chat/${chat.id}`} />}>
                                                    <span>{chat.title}</span>
                                                </SidebarMenuButton>
                                            </SidebarMenuItem>
                                        ))}
                                    </SidebarMenu>
                                </SidebarGroupContent>
                            </CollapsibleContent>
                        </SidebarGroup>
                    </Collapsible>
                </SidebarContent>
                <SidebarFooter>
                    <NavUser user={data.user} onSettingsClick={() => setSettingsOpen(true)} />
                </SidebarFooter>
            </Sidebar>

            <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Settings</DialogTitle>
                        <DialogDescription>
                            Manage your app preferences here.
                        </DialogDescription>
                    </DialogHeader>
                </DialogContent>
            </Dialog>
        </>
    )
}
