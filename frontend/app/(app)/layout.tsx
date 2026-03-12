import { redirect } from "next/navigation";

import { AppSidebar } from "@/components/app-sidebar";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import type { SidebarUser } from "@/hooks/use-sidebar-user";
import { getServerCurrentUser } from "@/lib/backend";

const toSidebarUser = (user: Awaited<ReturnType<typeof getServerCurrentUser>>): SidebarUser | null => {
  if (!user?.email?.trim()) {
    return null;
  }

  return {
    name: user.full_name?.trim() || user.email,
    email: user.email,
    avatar: user.avatar_url?.trim() || null,
  };
};

export default async function AppLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const currentUser = await getServerCurrentUser();
  const initialUser = toSidebarUser(currentUser);

  if (!initialUser) {
    redirect("/login");
  }

  return (
    <SidebarProvider className="h-svh max-h-svh overflow-hidden">
      <AppSidebar initialUser={initialUser} />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="flex h-12 shrink-0 items-center px-3">
          <SidebarTrigger />
        </header>
        <div className="min-h-0 min-w-0 flex-1 overflow-hidden p-2">{children}</div>
      </div>
    </SidebarProvider>
  );
}
