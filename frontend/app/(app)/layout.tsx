import { AppSidebar } from "@/components/app-sidebar";
import { ChatShell } from "@/components/chat-shell";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import type { SidebarUser } from "@/hooks/use-sidebar-user";
import { UserProvider } from "@/contexts/user-context";
import { getServerCurrentUser } from "@/lib/backend";

const PLACEHOLDER_EMAIL_DOMAIN = "users.local";

const toSidebarUser = (user: Awaited<ReturnType<typeof getServerCurrentUser>>): SidebarUser | null => {
  if (!user) return null;
  const username = (user as { username?: string | null }).username?.trim() ?? "";
  const email = user.email?.trim() ?? "";
  const isPlaceholder = email.endsWith(`@${PLACEHOLDER_EMAIL_DOMAIN}`);
  const subtitle = username ? `@${username}` : isPlaceholder ? "" : email;
  const displayId = username || email;
  if (!displayId) return null;

  return {
    name: user.full_name?.trim() || username || email,
    email: subtitle,
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

  return (
    <UserProvider user={initialUser}>
      <SidebarProvider className="h-svh max-h-svh overflow-hidden">
        <AppSidebar initialUser={initialUser} />
        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <header className="flex h-12 shrink-0 items-center px-3">
            <SidebarTrigger />
          </header>
          <div className="min-h-0 min-w-0 flex-1 overflow-hidden p-2">
            <ChatShell>{children}</ChatShell>
          </div>
        </div>
      </SidebarProvider>
    </UserProvider>
  );
}
