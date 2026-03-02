"use client"

import { useState } from "react"
import { usePathname } from "next/navigation"
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
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar"
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,

} from "@/components/ui/collapsible"
import { NavUser } from "@/components/nav-user"
import { SettingsDialog } from "@/components/settings-diag"
import { useKeyboardShortcut } from "@/hooks/use-keyboard-shortcut"
import { CHAT_HISTORY } from "@/lib/chat-threads"
import { ChevronRight, PlusCircle, SearchIcon } from "lucide-react"

const APP_USER = {
    name: "Test User",
    email: "test@example.com",
    avatar: "/avatars/shadcn.jpg",
}

export function AppSidebar() {
    const [settingsOpen, setSettingsOpen] = useState(false)
    const pathname = usePathname()

    useKeyboardShortcut(["ctrl", "shift", ","], () => setSettingsOpen((prev) => !prev))

    return (
        <>
            <Sidebar collapsible="icon">
                <SidebarHeader className="h-12" />
                <SidebarContent>
                    <SidebarGroup key="actions">
                        <SidebarGroupContent>
                            <SidebarMenu>
                                <SidebarMenuItem key="new-chat">
                                    <SidebarMenuButton
                                        render={<Link href="/" />}
                                    >
                                        <PlusCircle />
                                        <span>New chat</span>
                                    </SidebarMenuButton >
                                </SidebarMenuItem>
                                <SidebarMenuItem key="search-chat">
                                    <SidebarMenuButton render={<Link href="/" />}>
                                        <SearchIcon />
                                        <span>Search</span>
                                    </SidebarMenuButton >
                                </SidebarMenuItem>
                            </SidebarMenu>
                        </SidebarGroupContent>
                    </SidebarGroup>
                    <Collapsible defaultOpen className="group/collapsible">
                        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
                            <SidebarGroupLabel render={<CollapsibleTrigger className="group hover:cursor-pointer">
                                History
                                <ChevronRight className="ml-auto group-data-[panel-open]:rotate-90" />
                            </CollapsibleTrigger>}>
                            </SidebarGroupLabel>
                            <CollapsibleContent>
                                <SidebarGroupContent>
                                    <SidebarMenu>
                                        {CHAT_HISTORY.map((chat) => {
                                            const href = `/chats/${chat.id}`

                                            return (
                                                <SidebarMenuItem key={chat.id}>
                                                    <SidebarMenuButton
                                                        isActive={pathname === href}
                                                        render={<Link href={href} />}
                                                    >
                                                    <span>{chat.title}</span>
                                                    </SidebarMenuButton>
                                                </SidebarMenuItem>
                                            )
                                        })}
                                    </SidebarMenu>
                                </SidebarGroupContent>
                            </CollapsibleContent>
                        </SidebarGroup>
                    </Collapsible>
                </SidebarContent>
                <SidebarFooter>
                    <NavUser user={APP_USER} onSettingsClick={() => setSettingsOpen(true)} />
                </SidebarFooter>
            </Sidebar>
            <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
        </>
    )
}
