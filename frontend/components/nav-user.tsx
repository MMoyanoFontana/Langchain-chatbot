"use client"

import React from "react"
import {
    ChevronsUpDown,
    LogOut,
    MonitorIcon,
    MoonIcon,
    PaletteIcon,
    Settings,
    SunIcon,
} from "lucide-react"

import {
    Avatar,
    AvatarFallback,
    AvatarImage,
} from "@/components/ui/avatar"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuGroup,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuPortal,
    DropdownMenuRadioGroup,
    DropdownMenuRadioItem,
    DropdownMenuSeparator,
    DropdownMenuShortcut,
    DropdownMenuSub,
    DropdownMenuSubContent,
    DropdownMenuSubTrigger,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    useSidebar,
} from "@/components/ui/sidebar"
import { useTheme } from "next-themes"

export function NavUser({
    user,
    onSettingsClick,
}: {
    user: {
        name: string
        email: string
        avatar: string
    }
    onSettingsClick?: () => void
}) {
    const { isMobile } = useSidebar()
    const initials =
        user.name
            .split(" ")
            .filter(Boolean)
            .slice(0, 2)
            .map((part) => part[0]?.toUpperCase())
            .join("") || "U"

    const { theme, setTheme } = useTheme()


    return (
        <SidebarMenu>
            <SidebarMenuItem>
                <DropdownMenu>
                    <DropdownMenuTrigger
                        render={
                            <SidebarMenuButton
                                size="lg"
                                className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                            />
                        }
                    >
                        <Avatar className="h-8 w-8 rounded-lg">
                            <AvatarFallback className="rounded-lg">{initials}</AvatarFallback>
                            <AvatarImage src={user.avatar} alt={user.name} />
                        </Avatar>
                        <div className="grid flex-1 text-left text-sm leading-tight">
                            <span className="truncate font-medium">{user.name}</span>
                        </div>
                        <ChevronsUpDown className="ml-auto size-4" />
                    </DropdownMenuTrigger>
                    <DropdownMenuContent
                        className="w-(--anchor-width) min-w-56 rounded-lg"
                        side={isMobile ? "bottom" : "top"}
                        align="end"
                        sideOffset={5}
                    >
                        <DropdownMenuGroup>
                            <DropdownMenuLabel className="p-1 pb-2">
                                <div className="grid flex-1 text-left gap-1 px-1 py-1 text-left leading-tight">
                                    <span className="truncate text-sm">{user.email}</span>
                                </div>
                            </DropdownMenuLabel>
                        </DropdownMenuGroup>
                        <DropdownMenuGroup>
                            <DropdownMenuItem onClick={onSettingsClick}>
                                <Settings />
                                Settings
                                <DropdownMenuShortcut>Ctrl+⇧+,</DropdownMenuShortcut>
                            </DropdownMenuItem>
                            <DropdownMenuSub>
                                <DropdownMenuSubTrigger>
                                    <PaletteIcon />
                                    Theme
                                </DropdownMenuSubTrigger>
                                <DropdownMenuPortal>
                                    <DropdownMenuSubContent>
                                        <DropdownMenuGroup>
                                            <DropdownMenuLabel>Appearance</DropdownMenuLabel>
                                            <DropdownMenuRadioGroup
                                                value={theme}
                                                onValueChange={setTheme}
                                            >
                                                <DropdownMenuRadioItem value="light">
                                                    <SunIcon />
                                                    Light
                                                </DropdownMenuRadioItem>
                                                <DropdownMenuRadioItem value="dark">
                                                    <MoonIcon />
                                                    Dark
                                                </DropdownMenuRadioItem>
                                                <DropdownMenuRadioItem value="system">
                                                    <MonitorIcon />
                                                    System
                                                </DropdownMenuRadioItem>
                                            </DropdownMenuRadioGroup>
                                        </DropdownMenuGroup>
                                    </DropdownMenuSubContent>
                                </DropdownMenuPortal>
                            </DropdownMenuSub>
                        </DropdownMenuGroup>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem variant="destructive">
                            <LogOut />
                            Log out
                        </DropdownMenuItem>
                    </DropdownMenuContent>
                </DropdownMenu>
            </SidebarMenuItem>
        </SidebarMenu >
    )
}
