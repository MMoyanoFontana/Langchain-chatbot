export const PROVIDER_CODES = [
  "openai",
  "anthropic",
  "gemini",
  "groq",
  "ollama",
  "other",
] as const;

export type BackendProviderCode = (typeof PROVIDER_CODES)[number];

export const isBackendProviderCode = (
  value: string
): value is BackendProviderCode =>
  (PROVIDER_CODES as readonly string[]).includes(value);
