"use client";

import { useEffect, useState } from "react";

import type { BackendProviderCode } from "@/contexts/chat-composer-context";

export interface ChatCatalogProvider {
  id: number;
  code: string;
  display_name: string;
  is_active: boolean;
}

export interface ChatCatalogModelResponse {
  id: number;
  model_id: string;
  display_name: string;
  is_active: boolean;
  supports_reasoning?: boolean;
  provider: ChatCatalogProvider;
}

export interface ChatModel {
  id: string;
  name: string;
  providerLabel: string;
  providerSlug: string;
  providerCode: BackendProviderCode;
  supportsReasoning: boolean;
}

interface ProviderApiKeyResponse {
  provider?: { code?: string };
  is_active?: boolean;
}

const PROVIDER_LOGO_MAP: Record<string, string> = {
  gemini: "google",
};

export const normalizeProviderSlug = (providerCode: string): string =>
  PROVIDER_LOGO_MAP[providerCode] ?? providerCode;

export const normalizeProviderCode = (providerCode: string): BackendProviderCode => {
  if (
    providerCode === "openai" ||
    providerCode === "anthropic" ||
    providerCode === "gemini" ||
    providerCode === "groq" ||
    providerCode === "other"
  ) {
    return providerCode;
  }
  return "other";
};

export const mapCatalogModel = (entry: ChatCatalogModelResponse): ChatModel => ({
  id: entry.model_id,
  name: entry.display_name,
  providerLabel: entry.provider.display_name,
  providerSlug: normalizeProviderSlug(entry.provider.code),
  providerCode: normalizeProviderCode(entry.provider.code),
  supportsReasoning: Boolean(entry.supports_reasoning),
});

export interface UseChatModelsResult {
  models: ChatModel[];
  connectedProviderCodes: Set<BackendProviderCode>;
  loading: boolean;
}

/**
 * Fetches the model catalog and the user's connected provider keys.
 * Shared between the prompt composer (model picker) and the chat session
 * (compare-with picker on existing assistant messages).
 */
export function useChatModels(): UseChatModelsResult {
  const [models, setModels] = useState<ChatModel[]>([]);
  const [connectedProviderCodes, setConnectedProviderCodes] = useState<
    Set<BackendProviderCode>
  >(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const [modelsResponse, keysResponse] = await Promise.all([
          fetch("/api/models", { cache: "no-store" }),
          fetch("/api/user/provider-keys", { cache: "no-store" }),
        ]);

        if (!modelsResponse.ok) {
          throw new Error(`Models request failed with status ${modelsResponse.status}`);
        }
        if (!keysResponse.ok) {
          throw new Error(`Provider keys request failed with status ${keysResponse.status}`);
        }

        const [modelsPayload, keysPayload] = (await Promise.all([
          modelsResponse.json(),
          keysResponse.json(),
        ])) as [ChatCatalogModelResponse[], ProviderApiKeyResponse[]];

        if (cancelled) return;

        const mapped = modelsPayload.map(mapCatalogModel);
        const connected = new Set<BackendProviderCode>();
        for (const entry of keysPayload) {
          if (entry.is_active === false) continue;
          connected.add(
            normalizeProviderCode(entry.provider?.code?.toLowerCase() ?? "")
          );
        }

        setModels(mapped);
        setConnectedProviderCodes(connected);
      } catch {
        if (!cancelled) {
          setModels([]);
          setConnectedProviderCodes(new Set());
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return { models, connectedProviderCodes, loading };
}
