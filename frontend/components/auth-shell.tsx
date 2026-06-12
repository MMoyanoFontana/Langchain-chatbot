import Image from "next/image"
import { Suspense, type ComponentProps } from "react"
import { AuthPage } from "@/components/auth-page"

type AuthShellProps = {
  defaultMode?: ComponentProps<typeof AuthPage>["defaultMode"]
}

export function AuthShell({ defaultMode }: AuthShellProps) {
  return (
    <div className="flex min-h-svh items-center justify-center bg-muted p-6 md:p-10">
      <div className="flex w-full max-w-sm overflow-hidden rounded-xl shadow-sm md:max-w-3xl">
        <div className="flex-1 bg-card">
          <Suspense>
            <AuthPage defaultMode={defaultMode} />
          </Suspense>
        </div>

        {/* Logo panel stays white in both themes — the logo is dark-on-light. */}
        <div className="hidden flex-1 items-center justify-center bg-white p-6 md:flex">
          <Image src="/gemis.png" alt="Gemis" width={280} height={80} />
        </div>
      </div>
    </div>
  )
}
