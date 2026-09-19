import Link from "next/link"

export default function LegalLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="bg-muted min-h-svh">
      <div className="mx-auto flex min-h-svh max-w-3xl flex-col px-6 py-10 md:py-16">
        <header className="mb-10">
          <Link
            href="/"
            className="text-muted-foreground hover:text-foreground text-sm underline underline-offset-4"
          >
            &larr; Back to the app
          </Link>
        </header>

        <main className="bg-card text-card-foreground rounded-xl p-6 shadow-sm md:p-10">
          {children}
        </main>

        <footer className="text-muted-foreground mt-8 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
          <Link href="/privacy" className="underline underline-offset-4">
            Privacy Policy
          </Link>
          <Link href="/terms" className="underline underline-offset-4">
            Terms of Use
          </Link>
          <span className="ml-auto">
            A personal portfolio demo. Not a commercial service.
          </span>
        </footer>
      </div>
    </div>
  )
}
