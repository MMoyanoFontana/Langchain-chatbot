"use client"

import * as React from "react"
import {
  Save,
  Server,
  Settings2,
  KeyRound,
  MonitorIcon,
  MoonIcon,
  SunIcon,
  Trash2,
  UserRound,
} from "lucide-react"
import { useTheme } from "next-themes"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { ModelSelectorLogo } from "@/components/ai-elements/model-selector"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from "@/components/ui/select"
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
import { cn } from "@/lib/utils"

type NavItem = {
  id: SettingsTab
  name: string
  description: string
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>
}

type SettingsTab = "profile" | "providers" | "general"
type ThemeMode = "light" | "dark" | "system"
type Language = "en" | "es"
type BackendProviderCode = "openai" | "anthropic" | "gemini" | "groq" | "other"

type ProviderOption = {
  id: string
  name: string
  logo?: string
  inputType: "api-key" | "endpoint"
  providerCode: BackendProviderCode
}

type ProviderConnection = {
  id: string
  provider: string
  value: string
  inputType: "api-key" | "endpoint"
}

type ThemeOption = {
  value: ThemeMode
  label: string
  description: string
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>
}

type LanguageOption = {
  value: Language
  label: string
}

type StoredSettings = {
  language: Language
}

type CurrentUserResponse = {
  email: string
  full_name: string | null
}

type ProviderApiKeyResponse = {
  id: string
  key_name: string
  masked_api_key: string
  provider: {
    code: BackendProviderCode
  }
}

const SETTINGS_STORAGE_KEY = "langchain-chatbot.settings.v1"
const PROVIDER_OPTIONS: ProviderOption[] = [
  { id: "openai", name: "OpenAI", logo: "openai", inputType: "api-key", providerCode: "openai" },
  {
    id: "anthropic",
    name: "Anthropic",
    logo: "anthropic",
    inputType: "api-key",
    providerCode: "anthropic",
  },
  {
    id: "google-gemini",
    name: "Google Gemini",
    logo: "google",
    inputType: "api-key",
    providerCode: "gemini",
  },
  { id: "groq", name: "Groq", logo: "groq", inputType: "api-key", providerCode: "groq" },
  {
    id: "mistral",
    name: "Mistral",
    logo: "mistral",
    inputType: "api-key",
    providerCode: "other",
  },
  {
    id: "ollama",
    name: "Ollama (Local)",
    inputType: "endpoint",
    providerCode: "other",
  },
]

const THEME_OPTIONS: ThemeOption[] = [
  {
    value: "light",
    label: "Light",
    description: "Bright surfaces for daytime use.",
    icon: SunIcon,
  },
  {
    value: "dark",
    label: "Dark",
    description: "Lower glare in darker environments.",
    icon: MoonIcon,
  },
  {
    value: "system",
    label: "System",
    description: "Match your OS appearance setting.",
    icon: MonitorIcon,
  },
]

const LANGUAGE_OPTIONS: LanguageOption[] = [
  {
    value: "en",
    label: "English",
  },
  {
    value: "es",
    label: "Spanish",
  },
]

const nav: NavItem[] = [
  {
    id: "general",
    name: "General",
    description: "Manage default appearance and language preferences.",
    icon: Settings2,
  },
  {
    id: "profile",
    name: "Profile",
    description: "Update your personal details shown in the app.",
    icon: UserRound,
  },
  {
    id: "providers",
    name: "Providers",
    description: "Add API keys for cloud providers or connect local Ollama.",
    icon: KeyRound,
  },
]

const normalizeOllamaEndpoint = (value: string) => {
  const trimmed = value.trim()
  if (!trimmed) {
    return ""
  }
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed
  }
  return `http://${trimmed}`
}

export function SettingsDialog({
  open,
  onOpenChange,
  children,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  children?: React.ReactNode
}) {
  const { theme, setTheme } = useTheme()
  const [activeTab, setActiveTab] = React.useState<SettingsTab>("profile")
  const [displayName, setDisplayName] = React.useState("")
  const [email, setEmail] = React.useState("")
  const [language, setLanguage] = React.useState<Language>("en")
  const [providers, setProviders] = React.useState<ProviderConnection[]>([])
  const [providerDrafts, setProviderDrafts] = React.useState<Record<string, string>>({})
  const [storageReady, setStorageReady] = React.useState(false)
  const [profileReady, setProfileReady] = React.useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = React.useState(false)
  const [isDeletingUser, setIsDeletingUser] = React.useState(false)

  React.useEffect(() => {
    try {
      const raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY)
      if (raw) {
        const parsed = JSON.parse(raw) as Partial<StoredSettings>
        if (parsed.language === "en" || parsed.language === "es") {
          setLanguage(parsed.language)
        }
      }
    } catch {
      // Ignore malformed settings in local storage.
    } finally {
      setStorageReady(true)
    }
  }, [])

  const loadRemoteSettings = React.useCallback(async () => {
    try {
      const userResponse = await fetch("/api/user", { cache: "no-store" })
      if (userResponse.ok) {
        const user = (await userResponse.json()) as CurrentUserResponse
        setDisplayName(user.full_name?.trim() || "")
        setEmail(user.email ?? "")
      }
    } catch {
      // Ignore user bootstrap errors and keep defaults.
    } finally {
      setProfileReady(true)
    }

    try {
      const keysResponse = await fetch("/api/user/provider-keys", { cache: "no-store" })
      if (!keysResponse.ok) {
        return
      }

      const keys = (await keysResponse.json()) as ProviderApiKeyResponse[]
      const mappedProviders = keys
        .map((entry) => {
          const option = PROVIDER_OPTIONS.find(
            (provider) =>
              provider.id === entry.key_name && provider.providerCode === entry.provider.code
          )
          if (!option) {
            return null
          }

          return {
            id: option.id,
            provider: option.name,
            value: entry.masked_api_key,
            inputType: option.inputType,
          } satisfies ProviderConnection
        })
        .filter((entry): entry is ProviderConnection => entry !== null)

      setProviders(mappedProviders)
    } catch {
      // Ignore provider key bootstrap errors and keep empty state.
    }
  }, [])

  React.useEffect(() => {
    if (!open) {
      return
    }

    void loadRemoteSettings()
  }, [loadRemoteSettings, open])

  React.useEffect(() => {
    if (!storageReady) {
      return
    }

    try {
      window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({ language }))
    } catch {
      // Ignore write errors.
    }
  }, [language, storageReady])

  const syncProfile = React.useCallback(async () => {
    if (!profileReady) {
      return
    }

    const normalizedDisplayName = displayName.trim()
    const payload: { fullName: string | null; email?: string } = {
      fullName: normalizedDisplayName.length > 0 ? normalizedDisplayName : null,
    }
    const trimmedEmail = email.trim()
    if (trimmedEmail) {
      payload.email = trimmedEmail
    }

    try {
      await fetch("/api/user", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
    } catch {
      // Keep local edits even if sync fails.
    }
  }, [displayName, email, profileReady])

  const addProviderKey = async (provider: ProviderOption) => {
    const trimmed = (providerDrafts[provider.id] ?? "").trim()
    if (!trimmed) {
      return
    }

    const nextValue =
      provider.inputType === "api-key"
        ? trimmed
        : normalizeOllamaEndpoint(trimmed)

    if (!nextValue) {
      return
    }

    try {
      const response = await fetch(`/api/user/provider-keys/${provider.providerCode}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          apiKey: nextValue,
          keyName: provider.id,
          isDefault: true,
          isActive: true,
        }),
      })

      if (!response.ok) {
        return
      }

      const stored = (await response.json()) as ProviderApiKeyResponse
      const nextKey: ProviderConnection = {
        id: provider.id,
        provider: provider.name,
        value: stored.masked_api_key,
        inputType: provider.inputType,
      }

      setProviders((previous) => {
        const withoutCurrent = previous.filter((entry) => entry.id !== nextKey.id)
        return [nextKey, ...withoutCurrent]
      })
      setProviderDrafts((previous) => ({ ...previous, [provider.id]: "" }))
    } catch {
      // Keep existing state if backend request fails.
    }
  }

  const removeProviderKey = async (provider: ProviderOption) => {
    try {
      await fetch(`/api/user/provider-keys/${provider.providerCode}/${provider.id}`, {
        method: "DELETE",
      })
    } catch {
      // Keep local cleanup even if backend request fails.
    }

    setProviders((previous) => previous.filter((entry) => entry.id !== provider.id))
    setProviderDrafts((previous) => ({ ...previous, [provider.id]: "" }))
  }

  const deleteCurrentUser = React.useCallback(async () => {
    setIsDeletingUser(true)
    try {
      const response = await fetch("/api/user", { method: "DELETE" })
      if (!response.ok) {
        return
      }

      setDeleteDialogOpen(false)
      setProviderDrafts({})
      await loadRemoteSettings()
    } finally {
      setIsDeletingUser(false)
    }
  }, [loadRemoteSettings])

  const activeTabMeta = nav.find((item) => item.id === activeTab) ?? nav[0]
  const activeTheme: ThemeMode =
    theme === "light" || theme === "dark" || theme === "system" ? theme : "system"
  const selectedLanguageOption =
    LANGUAGE_OPTIONS.find((option) => option.value === language) ?? LANGUAGE_OPTIONS[0]

  const tabContent = (() => {
    if (activeTab === "profile") {
      return (
        <section className="max-w-3xl ">
          <div>
            <div className="border-b py-4">
              <label className="grid gap-2">
                <span className="text-sm">Display name</span>
                <Input
                  value={displayName}
                  onChange={(event) => setDisplayName(event.target.value)}
                  onBlur={() => {
                    void syncProfile()
                  }}
                />
              </label>
            </div>

            <div className="border-b py-4">
              <label className="grid gap-2">
                <span className="text-sm">Email</span>
                <Input
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  onBlur={() => {
                    void syncProfile()
                  }}
                />
              </label>
            </div>
          </div>

          <div className="py-4">
            <div className="flex justify-end">
              <Button
                type="button"
                variant="destructive"
                onClick={() => setDeleteDialogOpen(true)}
              >
                <Trash2 />
                Delete account
              </Button>
            </div>
          </div>
        </section>
      )
    }

    if (activeTab === "providers") {
      const connectedProviders = PROVIDER_OPTIONS.filter((provider) =>
        providers.some((entry) => entry.id === provider.id)
      )
      const notConnectedProviders = PROVIDER_OPTIONS.filter(
        (provider) => !providers.some((entry) => entry.id === provider.id)
      )

      const renderProviderRow = (provider: ProviderOption) => {
        const existingKey = providers.find((entry) => entry.id === provider.id)
        const hasSavedKey = !!existingKey

        return (
          <form
            key={provider.id}
            onSubmit={(event) => {
              event.preventDefault()
              if (!hasSavedKey) {
                void addProviderKey(provider)
              }
            }}
            className="grid gap-2 md:grid-cols-[170px_1fr_auto] md:items-end"
          >
            <div className="flex h-8 items-center gap-2">
              {provider.logo ? (
                <ModelSelectorLogo
                  provider={provider.logo}
                  className="size-4 rounded-full ring-1 ring-border"
                />
              ) : (
                <span className="bg-muted/30 flex size-4 items-center justify-center rounded-full ring-1 ring-border">
                  <Server className="size-2.5" />
                </span>
              )}
              <span className="text-sm font-medium">{provider.name}</span>
            </div>
            {hasSavedKey ? (
              <p className="text-muted-foreground bg-muted/30 flex h-8 items-center rounded-lg px-2.5 font-mono text-sm">
                {existingKey.value}
              </p>
            ) : (
              <Input
                type={provider.inputType === "api-key" ? "password" : "url"}
                autoComplete="off"
                placeholder={
                  provider.inputType === "api-key"
                    ? "Paste API key"
                    : "http://localhost:11434"
                }
                value={providerDrafts[provider.id] ?? ""}
                onChange={(event) =>
                  setProviderDrafts((previous) => ({
                    ...previous,
                    [provider.id]: event.target.value,
                  }))
                }
              />
            )}
            <div className="flex items-center gap-2 md:self-end">
              {hasSavedKey ? (
                <Button
                  type="button"
                  variant="outline"
                  size="icon-sm"
                  aria-label="Delete key"
                  title="Delete key"
                  onClick={() => {
                    void removeProviderKey(provider)
                  }}
                >
                  <Trash2 />
                </Button>
              ) : (
                <Button
                  type="submit"
                  variant="outline"
                  size="icon-sm"
                  aria-label="Save key"
                  title="Save key"
                >
                  <Save />
                </Button>
              )}
            </div>
          </form>
        )
      }

      return (
        <section className="max-w-3xl ">
          <div className="py-4">
            <h3 className="text-muted-foreground text-xs font-medium uppercase tracking-wide mb-2">
              Connected providers
            </h3>
            {connectedProviders.length > 0 ? (
              <div className="grid gap-2">
                {connectedProviders.map((provider) => renderProviderRow(provider))}
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">
                No connected providers yet.
              </p>
            )}
          </div>

          <div className="py-4">
            <h3 className="text-muted-foreground text-xs font-medium uppercase tracking-wide mb-2">
              Not connected
            </h3>
            {notConnectedProviders.length > 0 ? (
              <div className="grid gap-2">
                {notConnectedProviders.map((provider) => renderProviderRow(provider))}
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">
                All providers are connected.
              </p>
            )}
          </div>
        </section>
      )
    }

    if (activeTab === "general") {
      return (
        <section className="max-w-3xl">
          <div className="space-y-3 border-b py-4">
            <div className="space-y-1">
              <h3 className="text-sm font-medium">Appearance</h3>
              <p className="text-muted-foreground text-xs">
                Choose how the interface looks across the app.
              </p>
            </div>

            <div className="grid gap-2 sm:grid-cols-3">
              {THEME_OPTIONS.map((option) => {
                const Icon = option.icon
                const isActive = activeTheme === option.value

                return (
                  <button
                    key={option.value}
                    aria-pressed={isActive}
                    className={cn(
                      "flex w-full cursor-pointer flex-col gap-1.5 rounded-lg border px-3 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                      isActive
                        ? "border-foreground/20 bg-muted"
                        : "border-border hover:bg-muted/60"
                    )}
                    onClick={() => setTheme(option.value)}
                    type="button"
                  >
                    <span className="flex items-center gap-2 text-sm font-medium">
                      <Icon className="size-4 shrink-0" />
                      {option.label}
                    </span>
                    <span className="text-muted-foreground text-xs leading-relaxed">
                      {option.description}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>

          <div className="space-y-3 py-4">
            <div className="space-y-1">
              <h3 className="text-sm font-medium">Display language</h3>
              <p className="text-muted-foreground text-xs">
                Controls labels and other interface text.
              </p>
            </div>

            <Select
              value={language}
              onValueChange={(value) => {
                if (value === "en" || value === "es") {
                  setLanguage(value)
                }
              }}
            >
              <SelectTrigger className="w-full sm:max-w-xs">
                <span className="truncate">{selectedLanguageOption.label}</span>
              </SelectTrigger>
              <SelectContent align="start">
                {LANGUAGE_OPTIONS.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    <span>{option.label}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </section>
      )
    }

    return null
  })()

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="overflow-hidden p-0 md:h-[500px] md:w-[700px] md:max-w-[700px] lg:w-[800px] lg:max-w-[800px]">
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
                        <SidebarMenuItem key={item.id}>
                          <SidebarMenuButton
                            isActive={activeTab === item.id}
                            onClick={() => setActiveTab(item.id)}
                          >
                            <item.icon />
                            <span>{item.name}</span>
                          </SidebarMenuButton>
                        </SidebarMenuItem>
                      ))}
                    </SidebarMenu>
                  </SidebarGroupContent>
                </SidebarGroup>
              </SidebarContent>
            </Sidebar>

            <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
              <header className="flex min-h-16 shrink-0 flex-col justify-center gap-0.5 border-b px-4 ">
                <h2 className="text-sm font-medium">{activeTabMeta.name}</h2>
                <p className="text-muted-foreground text-xs">{activeTabMeta.description}</p>
              </header>

              <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4">
                {children ?? tabContent}
              </div>
            </main>
          </SidebarProvider>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogTitle>Delete user?</DialogTitle>
          <DialogDescription>
            This actions removes all your data and cannot be undone.
          </DialogDescription> 
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
              disabled={isDeletingUser}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant="destructive"
              onClick={() => {
                void deleteCurrentUser()
              }}
              disabled={isDeletingUser}
            >
              {isDeletingUser ? "Deleting..." : "Delete user"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
