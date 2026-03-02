"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import {
  Bell,
  Check,
  Globe,
  Home,
  Keyboard,
  Link as LinkIcon,
  Lock,
  Menu,
  MessageCircle,
  Paintbrush,
  Settings,
  Video,
} from "lucide-react"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
} from "@/components/ui/sidebar"

type NavItem = {
  name: string
  href: string
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>
}

const nav: NavItem[] = [
  { name: "Notifications", href: "/settings/notifications", icon: Bell },
  { name: "Navigation", href: "/settings/navigation", icon: Menu },
  { name: "Home", href: "/settings/home", icon: Home },
  { name: "Appearance", href: "/settings/appearance", icon: Paintbrush },
  { name: "Messages & media", href: "/settings/messages-media", icon: MessageCircle },
  { name: "Language & region", href: "/settings/language-region", icon: Globe },
  { name: "Accessibility", href: "/settings/accessibility", icon: Keyboard },
  { name: "Mark as read", href: "/settings/mark-as-read", icon: Check },
  { name: "Audio & video", href: "/settings/audio-video", icon: Video },
  { name: "Connected accounts", href: "/settings/connected-accounts", icon: LinkIcon },
  { name: "Privacy & visibility", href: "/settings/privacy-visibility", icon: Lock },
  { name: "Advanced", href: "/settings/advanced", icon: Settings },
]

export function SettingsDialog({
  open,
  onOpenChange,
  children,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  children?: React.ReactNode
}) {
  const pathname = usePathname()
  const router = useRouter()

  const handleOpenChange = (next: boolean) => {
    onOpenChange(next)
    if (!next) {
      try {
        router.back()
      } catch {
        router.push("/")
      }
    }
  }

  const isActive = (href: string) =>
    pathname === href || (pathname?.startsWith(href + "/") ?? false)

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="overflow-hidden p-0 md:max-h-[500px] md:max-w-[700px] lg:max-w-[800px]">
        <DialogTitle className="sr-only">Settings</DialogTitle>
        <DialogDescription className="sr-only">
          Customize your settings here.
        </DialogDescription>

        <SidebarProvider className="flex min-h-0 items-start">
          <Sidebar collapsible="none" className="hidden md:flex w-64">
            <SidebarContent>
              <SidebarGroup>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {nav.map((item) => (
                      <SidebarMenuItem key={item.href}>
                        <SidebarMenuButton
                          isActive={isActive(item.href)}
                          render={
                            <Link href={item.href} scroll={false} className="flex items-center gap-2">
                              <item.icon />
                              <span>{item.name}</span>
                            </Link>
                          }
                        />
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            </SidebarContent>
          </Sidebar>

          <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <header className="flex h-16 shrink-0 items-center gap-2 px-4" />

            <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4 pt-0">
              {children ?? (
                Array.from({ length: 10 }).map((_, i) => (
                  <div
                    key={i}
                    className="bg-muted/50 aspect-video max-w-3xl rounded-xl"
                  />
                ))
              )}
            </div>
          </main>
        </SidebarProvider>
      </DialogContent>
    </Dialog>
  )
}