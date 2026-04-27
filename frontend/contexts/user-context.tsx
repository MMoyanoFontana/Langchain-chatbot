"use client"

import { createContext, useContext } from "react"
import type { SidebarUser } from "@/hooks/use-sidebar-user"

type UserContextValue = {
  user: SidebarUser | null
}

const UserContext = createContext<UserContextValue>({ user: null })

export function UserProvider({
  user,
  children,
}: {
  user: SidebarUser | null
  children: React.ReactNode
}) {
  return <UserContext.Provider value={{ user }}>{children}</UserContext.Provider>
}

export function useCurrentUser(): UserContextValue {
  return useContext(UserContext)
}
