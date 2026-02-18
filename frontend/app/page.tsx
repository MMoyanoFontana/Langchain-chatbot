import ConversationDemo from "@/components/example";
import { AppSidebar } from "@/components/app-sidebar";
import {
	SidebarInset,
	SidebarProvider,
	SidebarTrigger,
} from "@/components/ui/sidebar";

export default function Page() {
	return (
		<SidebarProvider className="h-[100svh]">
			<AppSidebar />
			<SidebarInset className="min-h-0">
				<header className="flex h-12 items-center px-3">
					<SidebarTrigger />
				</header>
				<div className="min-h-0 flex-1 p-4">
					<ConversationDemo />
				</div>
				
			</SidebarInset>
		</SidebarProvider>
	);
}
