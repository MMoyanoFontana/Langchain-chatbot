"use client"

import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { useState } from "react"
import { AlertCircle } from "lucide-react"
import { z } from "zod"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldSeparator,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Spinner } from "@/components/ui/spinner"

type AuthMode = "login" | "signup"

type AuthPageProps = React.ComponentProps<"div"> & {
  defaultMode?: AuthMode
}

const loginSchema = z.object({
  username: z.string().trim().min(1, "Username is required."),
  password: z.string().min(1, "Password is required."),
})

const signupSchema = z.object({
  username: z
    .string()
    .trim()
    .min(3, "Username must be at least 3 characters.")
    .max(30, "Username must be 30 characters or fewer.")
    .regex(/^[a-zA-Z0-9_-]+$/, "Only letters, numbers, _ and - are allowed."),
  password: z.string().min(8, "Password must be at least 8 characters."),
  confirmPassword: z.string(),
  fullName: z.string().trim().max(120, "Full name must be 120 characters or fewer.").optional(),
}).refine((d) => d.password === d.confirmPassword, {
  message: "Passwords do not match.",
  path: ["confirmPassword"],
})

const parseApiError = async (response: Response, fallback: string) => {
  try {
    const payload = (await response.json()) as { error?: string }
    return payload.error?.trim() || fallback
  } catch {
    return fallback
  }
}

const verifySession = async () => {
  const response = await fetch("/api/user", { cache: "no-store", credentials: "include" })
  if (!response.ok) throw new Error(await parseApiError(response, "Session was not created."))
}

export function AuthPage({ className, defaultMode = "login", ...props }: AuthPageProps) {
  const searchParams = useSearchParams()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isRedirecting, setIsRedirecting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const returnTo = searchParams.get("returnTo")?.trim() || "/"
  const oauthError = searchParams.get("error")?.trim() || null
  const isSignup = defaultMode === "signup"
  const isBusy = isSubmitting || isRedirecting

  const submit = (formData: FormData) => {
    const username = String(formData.get("username") ?? "")
    const password = String(formData.get("password") ?? "")

    if (isSignup) {
      const fullName = String(formData.get("fullName") ?? "")
      const confirmPassword = String(formData.get("confirmPassword") ?? "")
      const parsed = signupSchema.safeParse({ username, password, confirmPassword, fullName })
      if (!parsed.success) {
        setError(parsed.error.issues[0]?.message ?? "Please check your input.")
        return
      }

      setIsSubmitting(true)
      setError(null)
      void (async () => {
        try {
          const response = await fetch("/api/auth/register", {
            body: JSON.stringify({
              username: parsed.data.username,
              fullName: parsed.data.fullName || null,
              password: parsed.data.password,
            }),
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            method: "POST",
          })
          if (!response.ok) { setError(await parseApiError(response, "Unable to create account.")); return }
          await verifySession()
          window.location.assign(returnTo)
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : "Unable to reach auth service.")
        } finally {
          setIsSubmitting(false)
        }
      })()
      return
    }

    const parsed = loginSchema.safeParse({ username, password })
    if (!parsed.success) {
      setError(parsed.error.issues[0]?.message ?? "Please check your input.")
      return
    }

    setIsSubmitting(true)
    setError(null)
    void (async () => {
      try {
        const response = await fetch("/api/auth/login", {
          body: JSON.stringify({ username: parsed.data.username, password: parsed.data.password }),
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          method: "POST",
        })
        if (!response.ok) { setError(await parseApiError(response, "Unable to sign in.")); return }
        await verifySession()
        window.location.assign(returnTo)
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Unable to reach auth service.")
      } finally {
        setIsSubmitting(false)
      }
    })()
  }

  const startOAuth = (provider: "google" | "github") => {
    if (isBusy) return
    setIsRedirecting(true)
    const params = new URLSearchParams({ returnTo })
    window.location.assign(`/api/auth/oauth/${provider}?${params.toString()}`)
  }

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card className="overflow-hidden p-0">
        <CardContent className="p-0">
          <form
            className="p-6 md:p-8"
            onSubmit={(e) => { e.preventDefault(); submit(new FormData(e.currentTarget)) }}
          >
            <FieldGroup>
              <div className="flex flex-col items-center gap-2 text-center">
                <h1 className="text-2xl font-bold">
                  {isSignup ? "Create an account" : "Welcome back"}
                </h1>
                <p className="text-balance text-muted-foreground">
                  {isSignup ? "Sign up to start chatting" : "Sign in to your account"}
                </p>
              </div>

              {(error || oauthError) && (
                <Alert variant="destructive">
                  <AlertCircle />
                  <AlertTitle>Authentication failed</AlertTitle>
                  <AlertDescription>{error || oauthError}</AlertDescription>
                </Alert>
              )}

              {isSignup && (
                <Field>
                  <FieldLabel htmlFor="fullName">Full name <span className="text-muted-foreground">(optional)</span></FieldLabel>
                  <Input id="fullName" name="fullName" type="text" placeholder="Your display name" disabled={isBusy} maxLength={120} />
                </Field>
              )}

              <Field>
                <FieldLabel htmlFor="username">Username</FieldLabel>
                <Input
                  id="username"
                  name="username"
                  type="text"
                  placeholder=""
                  required
                  disabled={isBusy}
                  autoComplete="username"
                  maxLength={30}
                />
              </Field>

              <Field>
                <FieldLabel htmlFor="password">Password</FieldLabel>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  required
                  minLength={isSignup ? 8 : 1}
                  disabled={isBusy}
                  autoComplete={isSignup ? "new-password" : "current-password"}
                />
              </Field>

              {isSignup && (
                <Field>
                  <FieldLabel htmlFor="confirmPassword">Confirm password</FieldLabel>
                  <Input id="confirmPassword" name="confirmPassword" type="password" required minLength={8} disabled={isBusy} autoComplete="new-password" />
                </Field>
              )}

              <Field>
                <Button type="submit" disabled={isBusy}>
                  {isSubmitting && <Spinner className="mr-1" />}
                  {isSignup ? "Sign up" : "Sign in"}
                </Button>
              </Field>

              <FieldSeparator className="*:data-[slot=field-separator-content]:bg-card">
                Or continue with
              </FieldSeparator>

              <Field className="grid grid-cols-2 gap-4">
                <Button variant="outline" type="button" disabled={isBusy} onClick={() => startOAuth("google")}>
                  {isRedirecting ? <Spinner className="mr-1" /> : (
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="size-4">
                      <path d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z" fill="currentColor" />
                    </svg>
                  )}
                  Google
                </Button>
                <Button variant="outline" type="button" disabled={isBusy} onClick={() => startOAuth("github")}>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" className="size-4">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8" fill="currentColor" />
                  </svg>
                  GitHub
                </Button>
              </Field>

              <FieldDescription className="text-center">
                {isSignup ? "Already have an account?" : "Don't have an account?"}{" "}
                <Link href={isSignup ? "/login" : "/signup"} className="underline underline-offset-4">
                  {isSignup ? "Sign in" : "Sign up"}
                </Link>
              </FieldDescription>
            </FieldGroup>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
