import ConversationDemo from "@/components/example";
import { AppSidebar } from "@/components/app-sidebar";
import {
	SidebarProvider,
	SidebarTrigger,
} from "@/components/ui/sidebar";

export default function Page() {
	return (
		<SidebarProvider className="h-[100svh]">
			<AppSidebar />
			<div className="flex flex-1 flex-col">
				<header className="flex h-12 items-center px-3">
					<SidebarTrigger />
				</header>
				<div className="min-h-0 flex-1 p-2">
					<ConversationDemo />
				</div>
			</div>

		</SidebarProvider>
	);
}


